import pandas as pd
import pickle
from tqdm import tqdm
import functions as fct
import numpy as np
from numba import jit
import parameters as par
import os
from more_itertools import consecutive_groups
import random


def anonymize(df_mmsr_secured, df_mmsr_unsecured, df_finrep, path=False):

    print("anonymize data")

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
    set_lei.update(
        set(pd.unique(df_finrep[["report_agent_lei"]].values.ravel("K")))
    )

    # allocate a random anonymised name to each lei
    anonymized_leis = random.sample(range(15000), len(set_lei))

    # modify the input databases with the ram
    df_lei = pd.DataFrame(
        index=list(set_lei), data={"anonymized_leis": anonymized_leis}
    )

    df_mmsr_unsecured = df_mmsr_unsecured.merge(
        df_lei, left_on="cntp_lei", right_index=True, how="left"
    )
    df_mmsr_unsecured.drop(columns="cntp_lei", inplace=True)
    df_mmsr_unsecured.rename(
        {"anonymized_leis": "cntp_lei"}, axis=1, inplace=True
    )
    df_mmsr_unsecured = df_mmsr_unsecured.merge(
        df_lei, left_on="report_agent_lei", right_index=True, how="left"
    )
    df_mmsr_unsecured.drop(columns="report_agent_lei", inplace=True)
    df_mmsr_unsecured.rename(
        {"anonymized_leis": "report_agent_lei"}, axis=1, inplace=True
    )

    df_mmsr_secured = df_mmsr_secured.merge(
        df_lei, left_on="cntp_lei", right_index=True, how="left"
    )
    df_mmsr_secured.drop(columns="cntp_lei", inplace=True)
    df_mmsr_secured.rename(
        {"anonymized_leis": "cntp_lei"}, axis=1, inplace=True
    )
    df_mmsr_secured = df_mmsr_secured.merge(
        df_lei, left_on="report_agent_lei", right_index=True, how="left"
    )
    df_mmsr_secured.drop(columns="report_agent_lei", inplace=True)
    df_mmsr_secured.rename(
        {"anonymized_leis": "report_agent_lei"}, axis=1, inplace=True
    )

    df_finrep = df_finrep.merge(
        df_lei, left_on="report_agent_lei", right_index=True, how="left"
    )
    df_finrep.drop(columns="report_agent_lei", inplace=True)
    df_finrep.rename(
        {"anonymized_leis": "report_agent_lei"}, axis=1, inplace=True
    )

    if path:

        # csv save
        os.makedirs(f"{path}pickle/", exist_ok=True)
        df_mmsr_secured.to_csv(f"{path}pickle/df_mmsr_secured.csv")
        df_mmsr_unsecured.to_csv(f"{path}pickle/df_mmsr_unsecured.csv")
        df_finrep.to_csv(f"{path}pickle/df_finrep.csv")

        # pickle save
        pickle.dump(
            df_mmsr_secured,
            open(f"{path}pickle/df_mmsr_secured.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        pickle.dump(
            df_mmsr_unsecured,
            open(f"{path}pickle/df_mmsr_unsecured.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        pickle.dump(
            df_finrep,
            open(f"{path}pickle/df_finrep.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return df_mmsr_secured, df_mmsr_unsecured, df_finrep


def reduce_size(df_mmsr_secured, df_mmsr_unsecured, path):

    print("reduce size")

    # reduce memory usage

    # for secured
    df_mmsr_secured.replace(
        {
            "trns_type": {
                "BORR": False,
                "LEND": True,
                "BUYI": False,
                "SELL": True,
            }
        },
        inplace=True,
    )
    df_mmsr_secured = df_mmsr_secured.astype(
        {
            "report_agent_lei": np.int16,
            "cntp_lei": np.int16,
            "trns_nominal_amt": np.float32,
        }
    )

    # for unsecured
    df_mmsr_unsecured.replace(
        {
            "trns_type": {
                "BORR": False,
                "LEND": True,
                "BUYI": False,
                "SELL": True,
            }
        },
        inplace=True,
    )
    df_mmsr_secured = df_mmsr_secured.astype(
        {
            "report_agent_lei": np.int16,
            "cntp_lei": np.int16,
            "trns_nominal_amt": np.float32,
        }
    )

    # save
    if path:
        os.makedirs(f"{path}pickle/", exist_ok=True)
        df_mmsr_secured.to_csv(f"{path}pickle/df_mmsr_secured.csv")
        df_mmsr_unsecured.to_csv(f"{path}pickle/df_mmsr_unsecured.csv")

        pickle.dump(
            df_mmsr_secured,
            open(f"{path}pickle/df_mmsr_secured.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        pickle.dump(
            df_mmsr_unsecured,
            open(f"{path}pickle/df_mmsr_unsecured.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def get_df_evergreen(df_mmsr_secured, flag_isin, sett_filter):

    print("get df_evergreen_clean")

    # option: select only the cases where settlement date ==  trade date but it doesn't work for the first line of the evergreen...
    if sett_filter:
        df_evergreen = df_mmsr_secured[
            df_mmsr_secured["settlement_date"] == df_mmsr_secured["trade_date"]
        ]
    else:
        df_evergreen = df_mmsr_secured

    # create a unique transaction id, as the one reported is not sufficient
    df_evergreen["line_id"] = df_evergreen.index

    # columns defining and evergreen
    columns = [
        "report_agent_lei",
        "cntp_lei",
        "tenor",
        "trns_nominal_amt",
    ]

    if flag_isin:
        columns = columns + ["coll_isin"]

    # group the lines by increasing start step
    df_evergreen_lists = df_evergreen.groupby(
        columns
        + [
            "start_step",
        ],
        as_index=False,
        dropna=False,
    ).agg({"line_id": tuple})

    # spot non consecutive dates
    breaks = ~df_evergreen_lists["start_step"].diff().isin([1, 2, 3, 4, 5])
    df_evergreen_lists["evergreen_group"] = breaks.cumsum()

    # restaure the initional line_id as index
    df_evergreen_explode = df_evergreen_lists.explode("line_id")
    df_evergreen_explode.set_index("line_id", inplace=True)

    # add the collumn evergreen group to the initial df
    df_evergreen = df_evergreen.merge(
        df_evergreen_explode["evergreen_group"],
        left_index=True,
        right_index=True,
    )

    # group the evergreen
    df_evergreen_clean = df_evergreen.groupby(
        columns + ["evergreen_group"],
        dropna=False,
    ).agg(
        start_steps=("start_step", lambda x: sorted(tuple(x))),
        evergreen=("start_step", lambda x: len(tuple(x)) > 1),
        line_id=("line_id", tuple),
        trade_date=("trade_date", min),
        maturity_date=("trade_date", max),
        start_step=("start_step", min),
        end_step=("start_step", max),
        trns_type=("trns_type", "last"),
        coll_isin=("coll_isin", "last"),
    )

    # reset the index of df_evergreen_clean
    if flag_isin:
        df_evergreen_clean.drop("coll_isin", axis=1, inplace=True)
    df_evergreen_clean = df_evergreen_clean.reset_index()

    # define the current tenor columnns as the notice period and build the true effective tenor
    df_evergreen_clean["notice_period"] = df_evergreen_clean["tenor"]
    df_evergreen_clean["tenor"] = (
        df_evergreen_clean["end_step"] - df_evergreen_clean["start_step"]
    )

    # get back the initial granularity of mmsr data base from the line_id
    df_evergreen = df_evergreen_clean.explode("line_id")
    df_evergreen.set_index("line_id", inplace=True)

    return df_evergreen, df_evergreen_clean


def get_df_expanded(
    df_clean,
    holidays=False,
    path=False,
    lending=True,
    var_name=False,
):
    """
    This function creates a dataframw where each contract is repeated on each line for each day it is active.
    """

    print("get df_mmsr_secured_expanded")

    # filter only on the reverse repo i.e. lending cash (except user choose the oposite)
    if lending:
        df = df_clean[df_clean["trns_type"]]
    else:
        df = df_clean[~df_clean["trns_type"]]
    df.drop("trns_type", axis=1, inplace=True)

    # get the max day from the max of the trade dates
    max_day = max(pd.to_datetime(df["trade_date"]))
    clipped_maturity_date = df["maturity_date"].clip(upper=max_day)

    # Create a list of dates for each contract
    if holidays:
        date_ranges = [
            pd.bdate_range(start, end, freq="C", holidays=holidays)
            for start, end in zip(df["trade_date"], clipped_maturity_date)
        ]
    else:
        date_ranges = [
            pd.date_range(start, end)
            for start, end in zip(df["trade_date"], clipped_maturity_date)
        ]

    # Duplicate rows based on date ranges
    df_expanded = df.loc[
        df.index.repeat([len(dates) for dates in date_ranges])
    ].copy()
    df_expanded["current_date"] = [
        date for dates in date_ranges for date in dates
    ]

    # Reset the index
    df_expanded.reset_index(drop=True, inplace=True)

    # save df_mmsr_secured_clean
    if path:
        df_expanded.to_csv(f"{path}pickle/{var_name}.csv")
        pickle.dump(
            df_expanded,
            open(f"{path}pickle/{var_name}.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return df_expanded


def get_df_mmsr_secured_expanded(
    df_mmsr_secured_clean,
    holidays=False,
    path=False,
):
    """
    This function creates a dataframw where each contract is repeated on each line for each day it is active.
    """

    print("get df_mmsr_secured_expanded")

    # filter only on the reverse repo i.e. lending cash
    df = df_mmsr_secured_clean[df_mmsr_secured_clean["trns_type"]]
    df.drop("trns_type", axis=1, inplace=True)

    # get the max day from the max of the trade dates
    max_day = max(pd.to_datetime(df["trade_date"]))
    clipped_maturity_date = df["maturity_date"].clip(upper=max_day)

    # Create a list of dates for each contract
    if holidays:
        date_ranges = [
            pd.bdate_range(start, end, freq="C", holidays=holidays)
            for start, end in zip(df["trade_date"], clipped_maturity_date)
        ]
    else:
        date_ranges = [
            pd.date_range(start, end)
            for start, end in zip(df["trade_date"], clipped_maturity_date)
        ]

    # Duplicate rows based on date ranges
    df_mmsr_secured_expanded = df.loc[
        df.index.repeat([len(dates) for dates in date_ranges])
    ].copy()
    df_mmsr_secured_expanded["current_date"] = [
        date for dates in date_ranges for date in dates
    ]

    # Reset the index
    df_mmsr_secured_expanded.reset_index(drop=True, inplace=True)

    # save df_mmsr_secured_clean
    if path:
        df_mmsr_secured_expanded.to_csv(
            f"{path}pickle/df_mmsr_secured_expanded.csv"
        )
        pickle.dump(
            df_mmsr_secured_expanded,
            open(f"{path}pickle/df_mmsr_secured_expanded.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return df_mmsr_secured_expanded


def get_dic_rev_repo_exp_adj_from_df_mmsr_secured_expanded(
    df_mmsr_secured_expanded, path=False, plot_period=False
):

    print("get dic_rev_repo_exp_adj from df_mmsr_secured_expanded")

    df = df_mmsr_secured_expanded.groupby(
        ["current_date", "report_agent_lei", "cntp_lei"]
    ).agg({"trns_nominal_amt": sum})

    days = df.index.get_level_values("current_date").unique()
    # max_day = max(pd.to_datetime(df_mmsr_secured_expanded["trade_date"]))
    # days = [day for day in days if day < max_day]

    leis = list(
        set(
            list(df.index.get_level_values("report_agent_lei"))
            + list(df.index.get_level_values("cntp_lei"))
        )
    )

    dic_rev_repo_exp_adj = {}
    for day in tqdm(days):
        dic_rev_repo_exp_adj.update(
            {day: pd.DataFrame(columns=leis, index=leis, data=0)}
        )
        df_rev_repo_exp = (
            df.loc[day].unstack("cntp_lei").droplevel(0, axis=1).fillna(0)
        )
        dic_rev_repo_exp_adj[day].loc[
            df_rev_repo_exp.index, df_rev_repo_exp.columns
        ] = df_rev_repo_exp

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
    else:
        plot_days = list(dic_rev_repo_exp_adj.keys())
        for day in plot_days:

            day_print = day.strftime("%Y-%m-%d")

            os.makedirs(
                f"{path}data/exposure_view/adj_matrices/weighted/",
                exist_ok=True,
            )
            fct.dump_np_array(
                dic_rev_repo_exp_adj[day],
                f"{path}data/exposure_view/adj_matrices/weighted/arr_reverse_repo_adj_{day_print}.csv",
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
                    f"{path}data/exposure_view/adj_matrices/{agg_period}/",
                    exist_ok=True,
                )
                fct.dump_np_array(
                    dic_arr_binary_adj[agg_period][step],
                    f"{path}data/exposure_view/adj_matrices/{agg_period}/arr_binary_adj_on_day_{day_print}.csv",
                )

    return dic_arr_binary_adj


def load_dic_arr_binary_adj(path):
    return pickle.load(open(f"{path}pickle/dic_arr_binary_adj.pickle", "rb"))


def build_arr_total_assets(df_finrep, path):

    # build the total asset per bank
    df_total_assets = df_finrep.set_index(
        ["qdate", "report_agent_lei"]
    ).unstack()
    arr_total_assets = np.array(df_total_assets)

    os.makedirs(path, exist_ok=True)
    df_total_assets.to_csv(f"{path}df_total_assets.csv")

    return arr_total_assets


def get_dic_dashed_trajectory(df_finrep, path=False):

    print("get dic_dashed_trajectory")

    dic_dashed_trajectory = {}
    plot_days = pd.to_datetime(
        sorted(list(set(df_finrep["qdate"].dt.strftime("%Y-%m-%d"))))
    )

    for day in plot_days:

        df_banks = (
            df_finrep[df_finrep["qdate"] == day]
            .set_index("report_agent_lei")
            .drop("qdate", axis=1)
        )
        dic_dashed_trajectory.update({day: df_banks})

        if path:
            day_print = day.strftime("%Y-%m-%d")
            os.makedirs(
                f"{path}data/accounting_view/dashed_trajectory/", exist_ok=True
            )
            df_banks.to_csv(
                f"{path}data/accounting_view/dashed_trajectory/df_banks_on_day_{day_print}.csv"
            )

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


def get_df_rev_repo_trans(df_mmsr_secured_clean, path=False):

    print("get df_rev_repo_trans")

    # rename the columns to match df_rev_repo_trans requirements (given by AB)
    dic_col_mapping = {
        "report_agent_lei": "owner_bank_id",
        "cntp_lei": "bank_id",
        "trns_nominal_amt": "amount",
    }
    df_rev_repo_trans = df_mmsr_secured_clean.rename(columns=dic_col_mapping)[
        [
            "owner_bank_id",
            "bank_id",
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

    # addapt the day to the lastest business day
    def app_func(row): return get_closest_bday(row["qdate"])

    df_finrep_clean["qdate"] = df_finrep_clean.apply(app_func, axis=1)

    if path:
        os.makedirs(f"{path}pickle/", exist_ok=True)
        df_finrep_clean.to_csv(f"{path}pickle/df_finrep_clean.csv")
        pickle.dump(
            df_finrep_clean,
            open(f"{path}pickle/df_finrep_clean.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return df_finrep_clean


def load_input_data_csv(path):

    df_mmsr_secured = pd.read_csv(
        f"{path}pickle/df_mmsr_secured.csv", index_col=0
    )
    df_mmsr_secured_clean = pd.read_csv(
        f"{path}pickle/df_mmsr_secured_clean.csv", index_col=0
    )
    df_mmsr_secured_expanded = pd.read_csv(
        f"{path}pickle/df_mmsr_secured_expanded.csv", index_col=0
    )

    df_mmsr_unsecured = pd.read_csv(
        f"{path}pickle/df_mmsr_unsecured.csv", index_col=0
    )
    df_finrep_clean = pd.read_csv(
        f"{path}pickle/df_finrep_clean.csv", index_col=0
    )

    for col in ["trade_date", "maturity_date"]:
        df_mmsr_secured[col] = pd.to_datetime(df_mmsr_secured[col])
        df_mmsr_secured_clean[col] = pd.to_datetime(df_mmsr_secured_clean[col])
        df_mmsr_secured_expanded[col] = pd.to_datetime(
            df_mmsr_secured_expanded[col]
        )
        df_mmsr_unsecured[col] = pd.to_datetime(df_mmsr_unsecured[col])

    df_mmsr_secured_expanded["current_date"] = pd.to_datetime(
        df_mmsr_secured_expanded["current_date"]
    )
    df_finrep_clean["qdate"] = pd.to_datetime(df_finrep_clean["qdate"])

    return (
        df_mmsr_secured,
        df_mmsr_secured_clean,
        df_mmsr_secured_expanded,
        df_mmsr_unsecured,
        df_finrep_clean,
    )


def load_input_data_pickle(path):

    df_mmsr_secured = pickle.load(
        open(f"{path}pickle/df_mmsr_secured.pickle", "rb")
    )
    df_mmsr_secured_clean = pickle.load(
        open(f"{path}pickle/df_mmsr_secured_clean.pickle", "rb")
    )
    if os.path.isfile(f"{path}pickle/df_expanded.pickle"):
        df_mmsr_secured_expanded = pickle.load(
            open(f"{path}pickle/df_expanded.pickle", "rb")
        )
    df_mmsr_unsecured = pickle.load(
        open(f"{path}pickle/df_mmsr_unsecured.pickle", "rb")
    )
    if os.path.isfile(f"{path}pickle/df_finrep_clean.pickle"):
        df_finrep_clean = pickle.load(
            open(f"{path}pickle/df_finrep_clean.pickle", "rb")
        )

    return (
        df_mmsr_secured,
        df_mmsr_secured_clean,
        df_mmsr_secured_expanded,
        df_mmsr_unsecured,
        df_finrep_clean,
    )


def add_ratios_in_df_finrep_clean(df_finrep_clean, path=False):
    columns = fct.list_exclusion(
        df_finrep_clean.columns,
        [
            "report_agent_lei",
            "qdate",
            "total assets",
            "riad_code",
            "entity_id",
            "signinst",
            "ulssmparent_riad_code",
            "date",
            "group_riad_code",
            "head",
            "ulssmparent_riad_code",
            "source",
        ],
    )

    for column in columns:
        df_finrep_clean[f"{column} over total assets"] = (
            df_finrep_clean[column] / df_finrep_clean["total assets"]
        )

    if path:
        df_finrep_clean.to_csv(f"{path}pickle/df_finrep_clean.csv")
        pickle.dump(
            df_finrep_clean,
            open(f"{path}pickle/df_finrep_clean.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
