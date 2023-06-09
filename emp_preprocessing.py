import pandas as pd
import pickle
from tqdm import tqdm
import functions as fct
import numpy as np
from numba import jit
import parameters as par
import os
from more_itertools import consecutive_groups
import data_mapping as dm
import random


def anonymize(df_mmsr_secured, df_mmsr_unsecured, df_finrep):
    # get the list of leis used within the input dataframes
    set_lei = set()
    set_lei.update(
        set(
            pd.unique(
                df_mmsr_secured[["cntp_lei", "report_agent_lei"]].values.ravel(
                    "K"
                )
            )
        )
    )
    set_lei.update(
        set(
            pd.unique(
                df_mmsr_unsecured[
                    ["cntp_lei", "report_agent_lei"]
                ].values.ravel("K")
            )
        )
    )
    set_lei.update(set(pd.unique(df_finrep[["lei"]].values.ravel("K"))))

    # allocate a random anonymised name to each lei
    anonymized_leis = random.sample(
        ["anonymized_" + str(i) for i in range(200)], len(set_lei)
    )
    dic_lei = dict(zip(set_lei, anonymized_leis))

    # modify the input databases with the ram
    df_mmsr_secured.replace(
        {"cntp_lei": dic_lei, "report_agent_lei": dic_lei}, inplace=True
    )
    df_mmsr_unsecured.replace(
        {"cntp_lei": dic_lei, "report_agent_lei": dic_lei}, inplace=True
    )
    df_finrep.replace({"lei": dic_lei}, inplace=True)


def get_df_mmsr_secured_clean(
    df_mmsr_secured, compute_tenor=False, holidays=False, path=False
):

    print("get df_mmsr_secured_clean")

    # ------------------------------------------
    # 1 - build the start_step and tenor columns

    # convert datetime to date
    df_mmsr_secured["trade_date_d"] = df_mmsr_secured["trade_date"].dt.date
    df_mmsr_secured["maturity_date_d"] = df_mmsr_secured[
        "maturity_date"
    ].dt.date

    if holidays:

        if compute_tenor:
            # get the tenor (in business days)
            lam_func = lambda row: np.busday_count(
                row["trade_date_d"], row["maturity_date_d"], holidays=holidays
            )
            df_mmsr_secured["tenor"] = df_mmsr_secured.apply(lam_func, axis=1)

        else:
            # get the tenor (in business days)
            df_mmsr_secured["tenor"] = df_mmsr_secured["maturity_band"]
            df_mmsr_secured.replace({"tenor": dm.dic_tenor}, inplace=True)

        # get the start_step (nb of the business day)
        df_mmsr_secured["first_date"] = df_mmsr_secured["trade_date_d"].min()
        lam_func = lambda row: np.busday_count(
            row["first_date"], row["trade_date_d"], holidays=holidays
        )
        df_mmsr_secured["start_step"] = df_mmsr_secured.apply(lam_func, axis=1)

    else:
        # get the tenor
        df_mmsr_secured["tenor"] = (
            df_mmsr_secured["maturity_date_d"]
            - df_mmsr_secured["trade_date_d"]
        ).dt.days

        # get the start_step
        df_mmsr_secured["first_date"] = df_mmsr_secured["trade_date_d"].min()
        df_mmsr_secured["start_step"] = (
            df_mmsr_secured["trade_date_d"] - df_mmsr_secured["first_date"]
        ).dt.days

    # drop unnecessary columns
    df_mmsr_secured.drop(
        columns=["trade_date_d", "maturity_date_d"], inplace=True
    )

    # ------------------------------------------
    # 2 - flag the evergreen repos

    # select only the columns common across an evergreen
    df_restricted = df_mmsr_secured[
        [
            "start_step",
            "report_agent_lei",
            "cntp_lei",
            "trns_nominal_amt",
            "tenor",
        ]
    ]

    # build a copy with start_step+1
    df_restricted_shift = df_restricted.copy()
    df_restricted_shift["start_step"] = df_restricted["start_step"] + 1

    # flags all the lines of an evergreen but the first one
    merged_df = df_restricted.merge(
        df_restricted_shift, indicator=True, how="left"
    )
    equal_rows = merged_df["_merge"] == "both"

    # the first line is obtained from the shifted copy (start_step+1)
    merged_df = df_restricted_shift.merge(
        df_restricted, indicator=True, how="left"
    )
    equal_rows_shift = merged_df["_merge"] == "both"

    # we take the logical OR between the 2 flags
    df_mmsr_secured["evergreen"] = equal_rows | equal_rows_shift

    # ------------------------------------------
    # 3 - set the start step as the min of each consecutive group of evergreens

    # select only the flagged evergreens
    df_evergreen = df_mmsr_secured[df_mmsr_secured["evergreen"]]

    # group the evergreen of same counterparties and amount
    df_evergreen_lists = df_evergreen.groupby(
        [
            "report_agent_lei",
            "cntp_lei",
            "trns_nominal_amt",
        ]
    ).agg(
        {
            "start_step": lambda x: list(x),
            "tenor": max,
            "unique_trns_id": lambda x: list(x),
            "maturity_date": max,
            "trade_date": min,
            "trns_type": max,  # used for filters
        }
    )
    df_evergreen_lists.rename(
        columns={"start_step": "list_start_steps"}, inplace=True
    )

    # find the min and max of each consecutive group of start steps
    min_in_cons_groups = lambda row: [
        min(group) for group in consecutive_groups(row["list_start_steps"])
    ]
    max_in_cons_groups = lambda row: [
        max(group) for group in consecutive_groups(row["list_start_steps"])
    ]
    df_evergreen_lists["min_start_steps"] = df_evergreen_lists.apply(
        min_in_cons_groups, axis=1
    )
    df_evergreen_lists["max_start_steps"] = df_evergreen_lists.apply(
        max_in_cons_groups, axis=1
    )

    # explode the min and max into new lines
    df_evergreen_clean = df_evergreen_lists.explode(
        ["min_start_steps", "max_start_steps"]
    )

    # reset the inedex
    df_evergreen_clean.reset_index(inplace=True)

    # define the effective observed tenor
    df_evergreen_clean["tenor"] = (
        df_evergreen_clean["tenor"]
        + df_evergreen_clean["max_start_steps"]
        - df_evergreen_clean["min_start_steps"]
    )
    df_evergreen_clean["start_step"] = df_evergreen_clean["min_start_steps"]

    # ------------------------------------------
    # 4 - build the df_mmsr_secured_clean

    # concatenate the evergreen and the other transactions
    df_mmsr_secured_clean = pd.concat(
        [
            df_mmsr_secured[df_mmsr_secured["evergreen"] == False],
            df_evergreen_clean,
        ]
    )

    # reset the index
    df_mmsr_secured_clean.reset_index(inplace=True, drop=True)

    # save df_mmsr_secured_clean
    if path:
        df_mmsr_secured_clean.to_csv(f"{path}pickle/df_mmsr_secured_clean.csv")

    return df_mmsr_secured_clean


def get_dic_rev_repo_exp_adj_from_mmsr_secured_clean(
    df_mmsr_secured_clean, path=False, plot_period=False
):

    print("get dic_rev_repo_exp_adj from df_mmsr_secured_clean")

    # filter only on the reverse repo i.e. lending cash
    df_mmsr_secured_clean = df_mmsr_secured_clean[
        df_mmsr_secured_clean["trns_type"].isin(["LEND", "SELL"])
    ]

    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(
        df_mmsr_secured_clean[["cntp_lei", "report_agent_lei"]].values.ravel(
            "K"
        )
    )

    # define the list of dates in the mmsr database
    days = pd.to_datetime(
        sorted(
            list(
                set(
                    df_mmsr_secured_clean["trade_date"].dt.strftime("%Y-%m-%d")
                )
            )
        )
    )

    # initialisation of a dictionary of the observed rev repo exposure adj
    dic_rev_repo_exp_adj = {}
    for day in pd.bdate_range(
        start=days[0],
        end=days[-1],
        freq="C",
        holidays=dm.holidays,
    ):
        dic_rev_repo_exp_adj.update(
            {day: pd.DataFrame(columns=leis, index=leis, data=0)}
        )

    # building of the matrices and storage in the dictionary observed_path (evergreen are already retreated in df_mmsr_clean)
    for index in tqdm(df_mmsr_secured_clean.index):
        for day in pd.bdate_range(
            start=df_mmsr_secured_clean.loc[index, "trade_date"],
            end=min(
                df_mmsr_secured_clean.loc[index, "maturity_date"],
                pd.to_datetime(days[-1]),
            ),
            freq="C",
            holidays=dm.holidays,
        ):
            dic_rev_repo_exp_adj[day].loc[
                df_mmsr_secured_clean.loc[index, "report_agent_lei"],
                df_mmsr_secured_clean.loc[index, "cntp_lei"],
            ] = (
                dic_rev_repo_exp_adj[day].loc[
                    df_mmsr_secured_clean.loc[index, "report_agent_lei"],
                    df_mmsr_secured_clean.loc[index, "cntp_lei"],
                ]
                + df_mmsr_secured_clean.loc[index, "trns_nominal_amt"]
            )

    # pickle dump
    if path:
        os.makedirs(f"{path}pickle/", exist_ok=True)
        pickle.dump(
            dic_rev_repo_exp_adj,
            open(f"{path}pickle/dic_rev_repo_exp_adj.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save to csv for the plot_days
    if plot_period:
        days = list(dic_rev_repo_exp_adj.keys())
        plot_days = fct.get_plot_days_from_period(days, plot_period)
        for day in plot_days:

            day_print = day.strftime("%Y-%m-%d")

            os.makedirs(
                f"{path}exposure_view/adj_matrices/weighted/",
                exist_ok=True,
            )
            fct.dump_np_array(
                dic_rev_repo_exp_adj[day],
                f"{path}exposure_view/adj_matrices/weighted/arr_reverse_repo_adj_{day_print}.csv",
            )

    return dic_rev_repo_exp_adj


def get_dic_rev_repo_exp_adj_from_exposures(
    df_exposures, path=False, plot_period=False
):
    print("get dic_rev_repo_exp_adj from exposure")

    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(df_exposures[["borr_lei", "lend_lei"]].values.ravel("K"))

    # define the list of dates in the mmsr database
    # mmsr_trade_dates = pd.unique(df_exposures["Setdate"])
    mmsr_trade_dates = pd.to_datetime(
        sorted(list(set(df_exposures["Setdate"].dt.strftime("%Y-%m-%d"))))
    )

    # initialisation of a dictionary of the observed paths
    dic_rev_repo_exp_adj = {}  # for the exposures
    for mmsr_trade_date in mmsr_trade_dates:
        dic_rev_repo_exp_adj.update(
            {mmsr_trade_date: pd.DataFrame(columns=leis, index=leis, data=0)}
        )

    # building of the matrices and storage in the dictionary observed_path
    for index in tqdm(df_exposures.index):
        ts_trade = pd.to_datetime(
            df_exposures.loc[index, "Setdate"].strftime("%Y-%m-%d")
        )
        dic_rev_repo_exp_adj[ts_trade].loc[
            df_exposures.loc[index, "lend_lei"],
            df_exposures.loc[index, "borr_lei"],
        ] = df_exposures.loc[index, "exposure"]

    # pickle dump
    if path:
        os.makedirs(f"{path}pickle/", exist_ok=True)
        pickle.dump(
            dic_rev_repo_exp_adj,
            open(f"{path}pickle/dic_rev_repo_exp_adj.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save to csv for the plot_days
    if plot_period:
        days = list(dic_rev_repo_exp_adj.keys())
        plot_days = fct.get_plot_days_from_period(days, plot_period)
        for day in plot_days:

            day_print = day.strftime("%Y-%m-%d")

            os.makedirs(
                f"{path}exposure_view/adj_matrices/weighted/",
                exist_ok=True,
            )
            fct.dump_np_array(
                dic_rev_repo_exp_adj[day],
                f"{path}exposure_view/adj_matrices/weighted/arr_reverse_repo_adj_{day_print}.csv",
            )

    return dic_rev_repo_exp_adj


def load_dic_rev_repo_exp_adj(path):
    return pickle.load(open(f"{path}pickle/dic_rev_repo_exp_adj.pickle", "rb"))


@jit(nopython=True)
def fast_build_arr_binary_adj(
    arr_obs_matrix_reverse_repo, arr_agg_period, nb_days, min_repo_trans_size=0
):

    # get the lenght of the arrays
    n, nb_banks, nb_banks = arr_obs_matrix_reverse_repo.shape
    n_agg_periods = arr_agg_period.shape[0]

    # tempory array of the aggregated ajency matrix (to compute the logical or)
    arr_temp_adj = np.zeros((n_agg_periods, nb_banks, nb_banks))

    # array of the aggregated adjency matrix
    arr_binary_adj = np.zeros(
        (n_agg_periods, nb_days + 1, nb_banks, nb_banks), dtype=np.int16
    )

    # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
    for day_nb in np.arange(nb_days + 1):

        # loop over the possible agg periods
        for period_nb in np.arange(n_agg_periods):

            # look backward all previous step in an agg period
            for agg_day_nb in np.arange(
                max(day_nb - arr_agg_period[period_nb], 0), day_nb
            ):

                # build a binary adjency matrix from the weighted adjency matrix
                binary_adj = np.where(
                    arr_obs_matrix_reverse_repo[agg_day_nb]
                    > min_repo_trans_size,
                    True,
                    False,
                )

                # update the tempory adj matrix to count the links
                arr_temp_adj[period_nb] = np.logical_or(
                    binary_adj, arr_temp_adj[period_nb]
                )

            # store the results in a dic for each agg period
            arr_binary_adj[period_nb, day_nb] = arr_temp_adj[period_nb]

            # reset the dic temporary to 0
            arr_temp_adj[period_nb] = np.zeros((nb_banks, nb_banks))

    return arr_binary_adj


def convert_dic_to_array(dic_obs_matrix_reverse_repo):
    # convert dic to array
    bank_ids = list(list(dic_obs_matrix_reverse_repo.values())[0].index)
    nb_banks = len(bank_ids)
    days = list(dic_obs_matrix_reverse_repo.keys())
    arr_obs_matrix_reverse_repo = np.fromiter(
        dic_obs_matrix_reverse_repo.values(),
        np.dtype((float, [nb_banks, nb_banks])),
    )
    return arr_obs_matrix_reverse_repo


def get_dic_arr_binary_adj(
    dic_rev_repo_exp_adj, path=False, plot_period=False
):

    print("get dic_arr_binary_adj")

    # convert dic to array
    bank_ids = list(list(dic_rev_repo_exp_adj.values())[0].index)
    nb_banks = len(bank_ids)
    days = list(dic_rev_repo_exp_adj.keys())
    arr_obs_matrix_reverse_repo = np.fromiter(
        dic_rev_repo_exp_adj.values(),
        np.dtype((float, [nb_banks, nb_banks])),
    )

    # convert list to array
    arr_agg_period = np.array(par.agg_periods)

    # build arr of results with numba
    arr_binary_adj = fast_build_arr_binary_adj(
        arr_obs_matrix_reverse_repo, arr_agg_period, len(days)
    )

    # convert array results to dictionaries
    dic_arr_binary_adj = {}
    for period_nb, agg_period in enumerate(par.agg_periods):
        dic_arr_binary_adj.update({agg_period: arr_binary_adj[period_nb]})

    # pickle dump
    if path:
        os.makedirs(f"{path}pickle/", exist_ok=True)
        pickle.dump(
            dic_arr_binary_adj,
            open(f"{path}pickle/dic_arr_binary_adj.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save to csv for the plot_days
    if plot_period:
        days = list(dic_rev_repo_exp_adj.keys())
        plot_steps = fct.get_plot_steps_from_period(days, plot_period)

        for agg_period in par.agg_periods:
            # save to csv for the plot_steps
            for step in plot_steps:
                day_print = days[step].strftime("%Y-%m-%d")
                os.makedirs(
                    f"{path}exposure_view/adj_matrices/{agg_period}/",
                    exist_ok=True,
                )
                fct.dump_np_array(
                    dic_arr_binary_adj[agg_period][step],
                    f"{path}exposure_view/adj_matrices/{agg_period}/arr_binary_adj_on_day_{day_print}.csv",
                )

    return dic_arr_binary_adj


def load_dic_arr_binary_adj(path):
    return pickle.load(open(f"{path}pickle/dic_arr_binary_adj.pickle", "rb"))


def build_arr_total_assets(df_finrep, path):

    # build the total asset per bank
    df_total_assets = df_finrep.set_index(["date", "lei"]).unstack()
    arr_total_assets = np.array(df_total_assets)

    os.makedirs(path, exist_ok=True)
    df_total_assets.to_csv(f"{path}df_total_assets.csv")

    return arr_total_assets


def get_dic_dashed_trajectory(df_finrep, path=False):

    print("get dic_dashed_trajectory")

    dic_dashed_trajectory = {}
    plot_days = pd.to_datetime(
        sorted(list(set(df_finrep["date"].dt.strftime("%Y-%m-%d"))))
    )

    for day in plot_days:
        df_banks = (
            df_finrep[df_finrep["date"] == plot_days[0]]
            .set_index("lei")
            .drop("date", axis=1)
        )
        dic_dashed_trajectory.update({day: df_banks})

    # pickle dump
    if path:
        os.makedirs(f"{path}pickle/", exist_ok=True)
        pickle.dump(
            dic_dashed_trajectory,
            open(f"{path}pickle/dic_dashed_trajectory.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return dic_dashed_trajectory


def load_dic_dashed_trajectory(path):
    return pickle.load(
        open(f"{path}pickle/dic_dashed_trajectory.pickle", "rb")
    )


def get_df_deposits_variations_by_bank(df_mmsr, dic_dashed_trajectory, path):

    print("get df_deposits_variations_by_bank")

    # filter only on the deposits instruments
    df_mmsr = df_mmsr[
        (df_mmsr["instr_type"] == "DPST")
        & df_mmsr["trns_type"].isin(["BORR", "BUYI"])
    ]

    # set the trade date as index
    df_mmsr.set_index("trade_date", inplace=True)

    # build the deposits time series (multi index bank and time)
    df_deposits = (
        df_mmsr.groupby("report_agent_lei")
        .resample("1d")
        .sum("trns_nominal_amt")
    )[["trns_nominal_amt"]]

    # get one column per bank
    df_deposits = df_deposits.unstack(0)
    df_deposits.columns = df_deposits.columns.droplevel()

    # compute the deposits variations (absolute and relative)
    df_delta_deposits = df_deposits.diff(1)
    df_relative_deposits = df_delta_deposits / df_deposits.shift(1)

    # select the bank ids that match df_finrep to build the variation over total assets
    df_banks = list(dic_dashed_trajectory.values())[
        -1
    ]  # take the last value of total assets (to be updated with something more fancy)
    bank_ids = fct.list_intersection(df_banks.index, df_deposits.columns)
    df_delta_deposits_over_assets = (
        df_delta_deposits[bank_ids] / df_banks["total assets"].T[bank_ids]
    )

    # rename all columns
    df_delta_deposits.columns = [
        f"{col} abs. var." for col in df_deposits.columns
    ]
    df_relative_deposits.columns = [
        f"{col} rel. var." for col in df_deposits.columns
    ]
    df_delta_deposits_over_assets.columns = [
        f"{col} var over tot. assets" for col in bank_ids
    ]

    # merge all data
    df_deposits_variations_by_bank = pd.merge(
        df_delta_deposits,
        df_relative_deposits,
        right_index=True,
        left_index=True,
    )
    df_deposits_variations_by_bank = pd.merge(
        df_deposits_variations_by_bank,
        df_delta_deposits_over_assets,
        right_index=True,
        left_index=True,
    )

    # save df_deposits_variations
    os.makedirs(path, exist_ok=True)
    df_deposits_variations_by_bank.to_csv(
        f"{path}df_deposits_variations_by_bank.csv"
    )

    return df_deposits_variations_by_bank


def get_df_deposits_variation(df_mmsr, dic_dashed_trajectory, path):

    print("get df_deposits_variation")

    # filter only on the deposits instruments
    df_mmsr = df_mmsr[
        (df_mmsr["instr_type"] == "DPST")
        & df_mmsr["trns_type"].isin(["BORR", "BUYI"])
    ]

    # set the trade date as index
    df_mmsr.set_index("trade_date", inplace=True)

    # build the deposits time series (multi index bank and time)
    df_deposits_variations = (
        df_mmsr.groupby("report_agent_lei")
        .resample("1d")
        .sum("trns_nominal_amt")
    )[["trns_nominal_amt"]]
    df_deposits_variations.rename(
        {"trns_nominal_amt": "deposits"}, axis=1, inplace=True
    )

    # compute the deposits variations (absolute and relative)
    df_deposits_variations["deposits abs. var."] = df_deposits_variations[
        "deposits"
    ].diff(1)
    df_deposits_variations["deposits rel. var."] = df_deposits_variations[
        "deposits abs. var."
    ] / df_deposits_variations["deposits"].shift(1)

    # select the bank ids that match df_finrep to build the variation over total assets
    df_banks = list(dic_dashed_trajectory.values())[
        -1
    ]  # take the last value of total assets (to be updated with something more fancy)
    df_banks.index.name = "report_agent_lei"

    df_deposits_variations = df_deposits_variations.join(
        df_banks[["total assets"]], how="inner"
    )

    df_deposits_variations["deposits var over tot. assets"] = (
        df_deposits_variations["deposits abs. var."]
        / df_deposits_variations["total assets"]
    )

    df_deposits_variations.drop(
        ["deposits", "total assets"], axis=1, inplace=True
    )

    # save df_deposits_variations
    os.makedirs(path, exist_ok=True)
    df_deposits_variations.to_csv(f"{path}df_deposits_variations.csv")

    return df_deposits_variations


def get_df_rev_repo_trans(df_mmsr_secured_clean, path=False):

    # filter only on the reverse repo i.e. lending cash
    df = df_mmsr_secured_clean[
        df_mmsr_secured_clean["trns_type"].isin(["LEND", "SELL"])
    ]

    # rename the columns to match df_rev_repo_trans requirements (given by AB)
    dic_col_mapping = {
        "report_agent_lei": "owner_bank_id",
        "cntp_lei": "bank_id",
        "unique_trns_id": "trans_id",
        "trns_nominal_amt": "amount",
    }
    df_rev_repo_trans = df.rename(columns=dic_col_mapping)[
        [
            "owner_bank_id",
            "bank_id",
            "trans_id",
            "amount",
            "start_step",
            "tenor",
        ]
    ]
    df_rev_repo_trans["status"] = False

    df_rev_repo_trans.reset_index(inplace=True, drop=True)

    if path:
        df_rev_repo_trans.to_csv(f"{path}pickle/df_rev_repo_trans.csv")

    return df_rev_repo_trans


def get_df_finrep_clean(df_finrep, path):

    df_finrep_clean = df_finrep.copy()

    app_func = lambda row: get_closest_bday(row["date"])

    df_finrep_clean["date"] = df_finrep_clean.apply(app_func, axis=1)

    if path:
        df_finrep_clean.to_csv(f"{path}pickle/df_finrep_clean.csv")

    return df_finrep_clean


def get_closest_bday(input_timestamp):
    bbday = pd.offsets.CustomBusinessDay(holidays=dm.holidays)
    return bbday.rollforward(input_timestamp)
