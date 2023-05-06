import numpy as np
import networkx as nx
import graphics as gx
import functions as fct
import pandas as pd
from tqdm import tqdm
import dask
import pickle
from tqdm import tqdm
import parameters as par


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
            # if it is in the end of the period, do:
            if step % agg_period == agg_period - 1:
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
            # otherwise, just extend the time series
            else:
                df_jaccard.loc[
                    day, f"jaccard index-{agg_period}"
                ] = df_jaccard.loc[
                    days[step - 1], f"jaccard index-{agg_period}"
                ]

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
            # if is in the end of the period
            if step % agg_period == agg_period - 1:
                df_density.loc[
                    day, f"network density-{agg_period}"
                ] = dic_arr_binary_adj[agg_period][step].sum() / (
                    nb_banks * (nb_banks - 1.0)
                )  # for a directed graph
            # otherwise, just extend the time series
            else:
                df_density.loc[
                    day, f"network density-{agg_period}"
                ] = df_density.loc[
                    days[step - 1], f"network density-{agg_period}"
                ]

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
                f"{path}arr_in_degree_{agg_period}.csv",
            )
            fct.dump_np_array(
                dic_out_degree[agg_period],
                f"{path}arr_out_degree_{agg_period}.csv",
            )
            fct.dump_np_array(
                dic_degree[agg_period], f"{path}arr_degree_{agg_period}.csv"
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


def build_df_banks_emp(df_finrep, day, path):

    # build the total asset per bank
    df_banks = df_finrep[df_finrep["date"] == day][["lei", "total assets"]]
    df_banks.set_index("lei", inplace=True)

    # save the results to csv
    df_banks.to_csv(f"{path}df_banks.csv")

    return df_banks
