import numpy as np
import networkx as nx
import functions as fct
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
import parameters as par
import cpnet
import os


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


def cpnet_test(bank_network, algo="BE"):
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


def gini(x):
    """
    This function computes the gini coeficient of a numpy arary.
    param: x: a numpy array
    return: the gini coeficient
    """
    total = 0
    for i, xi in enumerate(x):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def get_transaction_stats(df_trans, extension, days, path=False):

    print(f"get transaction stats{extension}")

    # initialisation
    df_transaction_stats = pd.DataFrame(
        index=days,
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days[1:]), 1):

        # repo transactions maturity av. network
        df_ending = df_trans[
            df_trans["tenor"] + df_trans["start_step"] == step - 1
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
        df_starting = df_trans[df_trans["start_step"] == step - 1]
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
