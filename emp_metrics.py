import numpy as np
import networkx as nx
import functions as fct
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
import parameters as par
import cpnet


def get_jaccard(dic_arr_binary_adj, days, path=False):

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
        fct.init_path(path)
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
        fct.init_path(path)
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
        fct.init_path(path)
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


def get_av_degree(dic_degree, days, path=False):

    print("get av. degree")

    # initialisation
    df_av_degree = pd.DataFrame(
        index=days,
        columns=[f"av. degree-{agg_period}" for agg_period in par.agg_periods],
    )

    # loop over the steps
    for step, day in enumerate(tqdm(days[1:]), 1):
        for agg_period in par.agg_periods:
            df_av_degree.loc[day, f"av. degree-{agg_period}"] = dic_degree[
                agg_period
            ][step].mean()

    if path:
        fct.init_path(path)
        df_av_degree.to_csv(f"{path}df_av_degree.csv")

    return df_av_degree


def build_arr_total_assets(df_finrep, bank_ids_int, path):

    # build the total asset per bank
    df_total_assets = (
        df_finrep[df_finrep["lei"].isin(bank_ids_int)]
        .set_index(["date", "lei"])
        .unstack()
    )
    arr_total_assets = np.array(df_total_assets)

    df_total_assets.to_csv(f"{path}df_total_assets.csv")

    return arr_total_assets


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
