import numpy as np
import networkx as nx
import functions as fct
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
import parameters as par
import cpnet
import os
import powerlaw
import graphics as gx


def get_rev_repo_exposure_stats(dic_arr_binary_adj, days, path=False):

    print("get jaccard")

    # initialisation
    df_jaccard = pd.DataFrame(
        index=days,
        columns=[
            f"jaccard index-{agg_period}" for agg_period in par.agg_periods
        ],
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days[1:]), 1):
        for agg_period in par.agg_periods:
            if step > agg_period:
                df_jaccard.loc[day, f"jaccard index-{agg_period}"] = (
                    np.logical_and(
                        dic_arr_binary_adj[agg_period][step],
                        dic_arr_binary_adj[agg_period][step - agg_period],
                    ).sum()
                    / np.logical_or(
                        dic_arr_binary_adj[agg_period][step],
                        dic_arr_binary_adj[agg_period][step - agg_period],
                    ).sum()
                )

    if path:
        os.makedirs(path, exist_ok=True)
        df_jaccard.to_csv(f"{path}df_jaccard.csv")

    return df_jaccard


def get_density(dic_arr_binary_adj, days, path=False):

    print("get density")

    # define variable
    nb_days, nb_banks, nb_banks = list(dic_arr_binary_adj.values())[0].shape

    # initialisation
    df_density = pd.DataFrame(
        index=days,
        columns=[
            f"network density-{agg_period}" for agg_period in par.agg_periods
        ],
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days[1:]), 1):
        for agg_period in par.agg_periods:
            df_density.loc[
                day, f"network density-{agg_period}"
            ] = dic_arr_binary_adj[agg_period][step].sum() / (
                nb_banks * (nb_banks - 1.0)
            )  # for a directed graph

    if path:
        os.makedirs(path, exist_ok=True)
        df_density.to_csv(f"{path}df_density.csv")

    return df_density


def get_degree_distribution(dic_arr_binary_adj, path=False):

    print("get degree distribution")

    # define variables
    nb_days, nb_banks, nb_banks = list(dic_arr_binary_adj.values())[0].shape

    # initialization
    dic_in_degree = {}
    dic_out_degree = {}
    dic_degree = {}
    for agg_period in par.agg_periods:
        dic_in_degree.update(
            {agg_period: np.zeros((nb_days, nb_banks), dtype=np.int16)}
        )
        dic_out_degree.update(
            {agg_period: np.zeros((nb_days, nb_banks), dtype=np.int16)}
        )
        dic_degree.update(
            {agg_period: np.zeros((nb_days, nb_banks), dtype=np.int16)}
        )

    for step in tqdm(range(1, nb_days)):
        # Build the degree distribution time series - version aggregated.
        for agg_period in par.agg_periods:

            # first define a networkx object.
            bank_network = nx.from_numpy_array(
                dic_arr_binary_adj[agg_period][step],
                parallel_edges=False,
                create_using=nx.DiGraph,
            )

            # fill in the arrays the dictionaries
            dic_in_degree[agg_period][step] = np.array(
                bank_network.in_degree(), dtype=np.int16
            )[:, 1]
            dic_out_degree[agg_period][step] = np.array(
                bank_network.out_degree(), dtype=np.int16
            )[:, 1]
            dic_degree[agg_period][step] = np.array(
                bank_network.degree(), dtype=np.int16
            )[:, 1]

    if path:
        os.makedirs(path, exist_ok=True)
        for agg_period in par.agg_periods:
            fct.dump_np_array(
                dic_in_degree[agg_period],
                f"{path}arr_in_degree_agg_{agg_period}.csv",
            )
            fct.dump_np_array(
                dic_out_degree[agg_period],
                f"{path}arr_out_degree_agg_{agg_period}.csv",
            )
            fct.dump_np_array(
                dic_degree[agg_period],
                f"{path}arr_degree_agg_{agg_period}.csv",
            )

    return dic_in_degree, dic_out_degree, dic_degree


def get_degree_stats(dic_degree, days, path=False):

    print("get degree stats")

    columns = [
        f"degree{extension}-{agg_period}"
        for extension in par.stat_extensions
        for agg_period in par.agg_periods
    ]

    # initialisation
    df_degree_stats = pd.DataFrame(
        index=days,
        columns=columns,
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days[1:]), 1):
        for agg_period in par.agg_periods:
            df_degree_stats.loc[
                day, f"degree av. network-{agg_period}"
            ] = dic_degree[agg_period][step].mean()
            df_degree_stats.loc[
                day, f"degree min network-{agg_period}"
            ] = dic_degree[agg_period][step].min()
            df_degree_stats.loc[
                day, f"degree max network-{agg_period}"
            ] = dic_degree[agg_period][step].max()

    if path:
        os.makedirs(path, exist_ok=True)
        df_degree_stats.to_csv(f"{path}df_degree_stats.csv")

    return df_degree_stats


def step_cpnet(bank_network, algo="BE"):
    if algo == "KM_ER":
        alg = cpnet.KM_ER()
    elif algo == "KM_config":
        alg = cpnet.KM_config()
    elif algo == "Divisive":
        alg = cpnet.Divisive()
    elif algo == "Rombach":
        alg = cpnet.Rombach()
    elif algo == "Rossa":
        alg = cpnet.Rossa()
    elif algo == "LapCore":
        alg = cpnet.LapCore()
    elif algo == "LapSgnCore":
        alg = cpnet.LapSgnCore()
    elif algo == "LowRankCore":
        alg = cpnet.LowRankCore()
    elif algo == "MINRES":
        alg = cpnet.MINRES()
    elif algo == "Surprise":
        alg = cpnet.Surprise()
    elif algo == "Lip":
        alg = cpnet.Lip()
    elif algo == "BE":
        alg = cpnet.BE()

    try:
        alg.detect(bank_network)  # Feed the network as an input
        x = alg.get_coreness()  # Get the coreness of nodes
        c = alg.get_pair_id()  # Get the group membership of nodes

        # Statistical significance test
        sig_c, sig_x, significant, p_value = cpnet.qstest(
            c,
            x,
            bank_network,
            alg,
            significance_level=0.05,
            # num_of_thread=1,
        )

    except:  # if one of the algorithm generates an error, fill all outputs with nan or empty list
        sig_c, sig_x, significant, p_value = {}, {}, [np.nan], [np.nan]

    return sig_c, sig_x, significant, p_value


def get_algo_cpnet(
    arr_reverse_repo_adj,
    algo,
    days,
    plot_period,
):

    print(f"core-periphery tests using the {algo} approach")

    # initialise results and path
    plot_steps = fct.get_plot_steps_from_period(days, plot_period)
    df_algo_cpnet = pd.DataFrame(
        {
            "cpnet sig_c": pd.Series([], dtype=object),
            "cpnet sig_x": pd.Series([], dtype=object),
            "cpnet significant": pd.Series([], dtype=object),
            "cpnet p-value": pd.Series([], dtype=float),
        }
    )

    for step in plot_steps:

        day = days[step]

        print(f"test on day {day}")

        # build nx object
        bank_network = nx.from_numpy_array(
            arr_reverse_repo_adj[step],
            parallel_edges=False,
            create_using=nx.DiGraph,
        )

        # run cpnet test
        sig_c, sig_x, significant, pvalues = step_cpnet(
            bank_network, algo=algo
        )

        df_algo_cpnet.loc[day] = [sig_c, sig_x, significant, pvalues[0]]

    return df_algo_cpnet


def get_cpnet(
    dic_arr_binary_adj,
    arr_rev_repo_exp_adj,
    days,
    plot_period,
    path=False,
):

    print("run core-periphery tests")

    # initialise results
    df_cpnet = pd.DataFrame(
        columns=[
            f"{col} {algo}-{agg_period}"
            for col in [
                "cpnet sig_c",
                "cpnet sig_x",
                "cpnet significant",
                "cpnet p-value",
            ]
            for agg_period in par.agg_periods
            for algo in par.cp_algos
        ],
    )

    for agg_period in list(dic_arr_binary_adj.keys()) + ["weighted"]:

        # case dijonction for the dictionary of adjency periods
        if agg_period == "weighted":
            arr_adj = arr_rev_repo_exp_adj
        else:
            arr_adj = dic_arr_binary_adj[agg_period]

        for algo in par.cp_algos:
            df_algo_cpnet = get_algo_cpnet(
                arr_adj,
                algo=algo,
                days=days,
                plot_period=plot_period,
            )

            for col in df_algo_cpnet.columns:
                df_cpnet[f"{col} {algo}-{agg_period}"] = df_algo_cpnet[col]

    if path:
        os.makedirs(path, exist_ok=True)
        df_cpnet.to_csv(f"{path}df_cpnet.csv")

    return df_cpnet


def fig_gini(x):
    """
    This function computes the gini coeficient of a numpy arary.
    param: x: a numpy array
    return: the gini coeficient
    """
    total = 0
    for i, xi in enumerate(x):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def get_transaction_stats(df_rev_repo_trans, extension, days, path=False):

    print(f"get transaction stats{extension}")

    # initialisation
    df_transaction_stats = pd.DataFrame(
        index=days,
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days[1:]), 1):

        # repo transactions maturity av. network
        df_ending = df_rev_repo_trans[
            df_rev_repo_trans["tenor"] + df_rev_repo_trans["start_step"]
            == step - 1
        ]
        if df_ending["amount"].sum() > 0:
            df_transaction_stats.loc[
                day, f"repo transactions maturity{extension}"
            ] = (df_ending["amount"] @ df_ending["tenor"]) / df_ending[
                "amount"
            ].sum()
        else:
            df_transaction_stats.loc[
                day, f"repo transactions maturity{extension}"
            ] = 0

        # repo transactions notional av. network
        df_starting = df_rev_repo_trans[
            df_rev_repo_trans["start_step"] == step - 1
        ]
        nb_trans = len(df_starting)
        if nb_trans > 0:
            df_transaction_stats.loc[
                day,
                f"repo transactions notional{extension}",
            ] = (df_starting["amount"].sum()) / nb_trans
        else:
            df_transaction_stats.loc[
                day,
                f"repo transactions notional{extension}",
            ] = 0

        # number repo transactions av. network
        df_transaction_stats.loc[
            day,
            f"number repo transactions{extension}",
        ] = nb_trans

    if path:
        os.makedirs(path, exist_ok=True)
        df_transaction_stats.to_csv(f"{path}df_transaction_stats.csv")

    return df_transaction_stats


def get_exposure_stats(arr_rev_repo_exp_adj, days, path=False):

    print("get exposure stats")

    # initialisation
    df_exposures_stats = pd.DataFrame(
        index=days,
        columns=[
            f"repo exposures{extension}"
            for extension in [" min network", " max network", " av. network"]
        ],
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days)):
        ar_rev_repo_exp_adj_non_zero = arr_rev_repo_exp_adj[step][
            np.nonzero(arr_rev_repo_exp_adj[step])
        ]
        if len(ar_rev_repo_exp_adj_non_zero) > 0:
            df_exposures_stats.loc[day, "repo exposures min network"] = np.min(
                ar_rev_repo_exp_adj_non_zero
            )
            df_exposures_stats.loc[day, "repo exposures max network"] = np.max(
                ar_rev_repo_exp_adj_non_zero
            )
            df_exposures_stats.loc[
                day, "repo exposures av. network"
            ] = np.mean(ar_rev_repo_exp_adj_non_zero)

    if path:
        os.makedirs(path, exist_ok=True)
        df_exposures_stats.to_csv(f"{path}df_exposures_stats.csv")

    return df_exposures_stats


def step_item_powerlaw(sr_data):

    # at least 2 data points required with non negligeable size
    if (len(sr_data.dropna()) > 1) and (sr_data.abs().sum() > par.float_limit):

        # fit the data with the powerlaw librairy
        powerlaw_fit = powerlaw.Fit(sr_data.dropna())

        # retrieve the alpha and p-value
        powerlaw_alpha = powerlaw_fit.power_law.alpha
        R, powerlaw_pvalue = powerlaw_fit.distribution_compare(
            "power_law", "exponential"
        )

    # fill with nan otherwise
    else:
        powerlaw_fit = np.nan
        powerlaw_alpha = np.nan
        powerlaw_pvalue = np.nan

    fits = [powerlaw_fit, powerlaw_alpha, powerlaw_pvalue]

    return fits


def get_powerlaw(
    dic_dashed_trajectory,
    days,
    plot_period,
    plot_days=False,
    path=False,
):

    print("run power law tests")

    # initialise results
    df_powerlaw = pd.DataFrame(
        columns=[
            f"{metric} {bank_item}"
            for metric in [
                "powerlaw fit",
                "powerlaw alpha",
                "powerlaw p-value",
            ]
            for bank_item in list(dic_dashed_trajectory.values())[0].columns
        ],
    )

    # get the list of ploting days
    if not (plot_days):
        plot_days = fct.get_plot_days_from_period(days, plot_period)

    for day in plot_days:

        # get df_banks for this day
        df_banks = dic_dashed_trajectory[day]

        # replace values smaller than 0 by nan (avoid powerlaw warnings)
        df = df_banks.mask(df_banks <= 0)

        for bank_item in df.columns:
            fits = step_item_powerlaw(df[bank_item])
            df_powerlaw.loc[day, f"powerlaw fit {bank_item}"] = fits[0]
            df_powerlaw.loc[day, f"powerlaw alpha {bank_item}"] = fits[1]
            df_powerlaw.loc[day, f"powerlaw p-value {bank_item}"] = fits[2]

    if path:
        os.makedirs(path, exist_ok=True)
        df_powerlaw.to_csv(f"{path}df_powerlaw.csv")

    return df_powerlaw


def run_n_plot_powerlaw(df, path):

    df_powerlaw = pd.DataFrame(
        index=df.columns,
        columns=["powerlaw alpha", "powerlaw p-value"],
    )

    # loop over the banks
    for col in df.columns:

        # get the fits
        fits = step_item_powerlaw(df[col])
        df_powerlaw.loc[col] = fits[1:]

        # plot
        if not (np.isnan(fits[2])):
            gx.plot_step_item_powerlaw(
                fits[0],
                fits[1],
                fits[2],
                col,
                path,
                auto_xlabel=False,
            )

    os.makedirs(path, exist_ok=True)
    df_powerlaw.to_csv(f"{path}df_powerlaw.csv")


def get_df_deposits(df_mmsr_unsecured, dic_dashed_trajectory):

    print("get df_deposits")

    # filter only on the deposits instruments
    df_mmsr_unsecured = df_mmsr_unsecured[
        (df_mmsr_unsecured["instr_type"] == "DPST")
        & df_mmsr_unsecured["trns_type"].isin(["BORR", "BUYI"])
    ]

    # build the deposits time series (multi index bank and time)
    df_deposits = (
        df_mmsr_unsecured.groupby(["report_agent_lei", "trade_date"]).sum(
            "trns_nominal_amt"
        )
    )[["trns_nominal_amt"]]

    # get one column per bank
    df_deposits = df_deposits.unstack(
        0
    )  # the unstack ensure the full list of business is in the index (as at least one bank as a depositis positive on each day)
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

    return (
        df_delta_deposits,
        df_relative_deposits,
        df_delta_deposits_over_assets,
    )


def get_df_deposits_variations_by_bank(
    df_delta_deposits,
    df_relative_deposits,
    df_delta_deposits_over_assets,
    path,
):

    print("get df_deposits_variations_by_bank")

    # rename all columns
    df_delta_deposits.rename(
        columns={col: f"{col} abs. var." for col in df_delta_deposits.columns},
        inplace=True,
    )
    df_relative_deposits.rename(
        columns={
            col: f"{col} rel. var." for col in df_relative_deposits.columns
        },
        inplace=True,
    )
    df_delta_deposits_over_assets.rename(
        columns={
            col: f"{col} var over tot. assets"
            for col in df_delta_deposits_over_assets.columns
        },
        inplace=True,
    )

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


def get_df_deposits_variations(
    df_delta_deposits,
    df_relative_deposits,
    df_delta_deposits_over_assets,
    path,
):

    print("get df_deposits_variation")

    # build df_deposits variations
    df_deposits_variations = pd.DataFrame()
    df_deposits_variations["deposits abs. var."] = df_delta_deposits.stack(
        dropna=False
    )
    df_deposits_variations["deposits rel. var."] = df_relative_deposits.stack(
        dropna=False
    )
    df_deposits_variations[
        "deposits var over tot. assets"
    ] = df_delta_deposits_over_assets.stack(dropna=False)

    # save df_deposits_variations
    os.makedirs(path, exist_ok=True)
    df_deposits_variations.to_csv(f"{path}df_deposits_variations.csv")

    return df_deposits_variations


def get_df_isin(df_mmsr_secured_expanded, path=False):

    df = df_mmsr_secured_expanded.reset_index(drop=False, inplace=False)

    df_isin = df.groupby(["current_date", "coll_isin"]).agg(
        {"trns_nominal_amt": "count"}
    )

    # save file
    if path:
        os.makedirs(f"{path}collateral_reuse/", exist_ok=True)
        df_isin.to_csv(f"{path}collateral_reuse/df_isin.csv")
    return df_isin
