import numpy as np
import networkx as nx
import graphics as gx
import functions as fct
import pandas as pd
from tqdm import tqdm
import dask
import pickle


def get_jaccard(dic_dic_binary_adj):
    """
    jaccard index of the transactions: better approach to mesure the stability of trading relationships when maturities are longer than one day
    """

    # define the lenght
    days = list(list(dic_dic_binary_adj.values())[0].keys())
    agg_periods = dic_dic_binary_adj.keys()

    # initialisation
    df_jaccard = pd.DataFrame(index=days, columns=agg_periods)

    # loop over the steps
    for step, day in enumerate(days[1:], 1):
        for agg_period in agg_periods:
            # if it is in the end of the period, do:
            if step % agg_period == agg_period - 1:
                df_jaccard.loc[day, agg_period] = (
                    np.logical_and(
                        dic_dic_binary_adj[agg_period][day],
                        dic_dic_binary_adj[agg_period][
                            days[step - agg_period]
                        ],
                    ).sum()
                    / np.logical_or(
                        dic_dic_binary_adj[agg_period][day],
                        dic_dic_binary_adj[agg_period][
                            days[step - agg_period]
                        ],
                    ).sum()
                )
            # otherwise, just extend the time series
            else:
                df_jaccard.loc[day, agg_period] = df_jaccard.loc[
                    days[step - 1], agg_period
                ]

    return df_jaccard


def get_density(dic_dic_binary_adj):

    # define variable
    days = list(list(dic_dic_binary_adj.values())[0].keys())
    agg_periods = dic_dic_binary_adj.keys()
    n_banks = len(list(list(dic_dic_binary_adj.values())[0].values())[0])

    # initialisation
    df_density = pd.DataFrame(index=days, columns=agg_periods)

    # loop over the steps
    for step, day in enumerate(days[1:], 1):
        for agg_period in agg_periods:
            # if is in the end of the period
            if step % agg_period == agg_period - 1:
                df_density.loc[day, agg_period] = dic_dic_binary_adj[
                    agg_period
                ][day].sum() / (
                    n_banks * (n_banks - 1.0)
                )  # for a directed graph
            # otherwise, just extend the time series
            else:
                df_density.loc[day, agg_period] = df_density.loc[
                    days[step - 1], agg_period
                ]

    return df_density


def get_degree_distribution(dic_dic_binary_adj, bank_ids):

    # define variables
    days = list(list(dic_dic_binary_adj.values())[0].keys())
    agg_periods = dic_dic_binary_adj.keys()

    # initialisation
    df_in_degree_distribution = pd.DataFrame(
        index=days, columns=bank_ids, dtype=float
    )
    df_out_degree_distribution = pd.DataFrame(
        index=days, columns=bank_ids, dtype=float
    )

    dic_in_degree = {}
    dic_out_degree = {}
    for agg_period in agg_periods:
        dic_in_degree.update({agg_period: df_in_degree_distribution.copy()})
        dic_out_degree.update({agg_period: df_out_degree_distribution.copy()})

    for step, day in enumerate(days[1:], 1):
        # Build the degree distribution time series - version aggregated.
        for agg_period in agg_periods:

            # if is in the end of the period
            if step % agg_period == agg_period - 1:

                # first define a networkx object.
                bank_network = nx.from_numpy_array(
                    dic_dic_binary_adj[agg_period][day],
                    parallel_edges=False,
                    create_using=nx.DiGraph,
                )
                # build an array of the in_degree per bank
                ar_in_degree = np.array(bank_network.in_degree(), dtype=float)[
                    :, 1
                ]
                ar_out_degree = np.array(
                    bank_network.out_degree(), dtype=float
                )[:, 1]

                dic_in_degree[agg_period].loc[day] = ar_in_degree
                dic_out_degree[agg_period].loc[day] = ar_out_degree

            # otherwise, just extend the time series
            else:
                dic_in_degree[agg_period].loc[day] = dic_in_degree[
                    agg_period
                ].loc[days[step - 1]]
                dic_out_degree[agg_period].loc[day] = dic_out_degree[
                    agg_period
                ].loc[days[step - 1]]

    return dic_in_degree, dic_out_degree


def build_df_banks(df_finrep, dic_in_degree, dic_out_degree, path):
    # select the final day (common between the 2 lists)
    mmsr_days = list(list(dic_in_degree.values())[0].index)
    finrep_days = list(df_finrep["date"])
    day = fct.last_common_element(mmsr_days, finrep_days)

    # build the degree per bank
    bank_ids = list(list(dic_in_degree.values())[0].columns)
    agg_periods = list(dic_in_degree.keys())
    df_banks = pd.DataFrame(index=bank_ids)
    for agg_period in agg_periods:
        df_banks[f"degree_{agg_period}"] = (
            dic_in_degree[agg_period].loc[day]
            + dic_out_degree[agg_period].loc[day]
        )

    # build the total asset per bank
    df_finrep = df_finrep[df_finrep["date"] == day]
    df_banks = df_banks.merge(
        df_finrep[["lei", "total assets"]], right_on="lei", left_index=True
    )
    df_banks.set_index("lei", inplace=True)

    # save the results to csv
    df_banks.to_csv(f"{path}df_banks.csv")

    return df_banks
