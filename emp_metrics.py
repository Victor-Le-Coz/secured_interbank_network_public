import numpy as np
import networkx as nx
import graphics as gx
import functions as fct
import pandas as pd


def get_jaccard(dic_binary_adjs):
    """
    jaccard index of the transactions: better approach to mesure the stability of trading relationships when maturities are longer than one day
    """

    # define the lenght
    nb_steps = len(list(dic_binary_adjs.values())[0])
    agg_periods = dic_binary_adjs.keys()

    # initialisation
    df_jaccard = pd.DataFrame(index=range(nb_steps), columns=agg_periods)

    # loop over the steps
    for step in range(1, nb_steps):
        for agg_period in agg_periods:
            # if it is in the end of the period, do:
            if step % agg_period == agg_period - 1:
                df_jaccard.loc[step, agg_period] = (
                    np.logical_and(
                        dic_binary_adjs[agg_period][step],
                        dic_binary_adjs[agg_period][step - agg_period],
                    ).sum()
                    / np.logical_or(
                        dic_binary_adjs[agg_period][step],
                        dic_binary_adjs[agg_period][step - agg_period],
                    ).sum()
                )
            # otherwise, just extend the time series
            else:
                df_jaccard.loc[step, agg_period] = df_jaccard.loc[
                    step - 1, agg_period
                ]

    return df_jaccard


def get_density(dic_binary_adjs):

    n_banks = len(list(dic_binary_adjs.values())[0][0])

    # initialisation
    dic_density = {}
    for agg_period in dic_binary_adjs.keys():
        dic_density.update({agg_period: [0.0]})

    # loop over the steps
    for step in range(1, len(list(dic_binary_adjs.values())[0])):
        for agg_period in dic_binary_adjs.keys():
            # if is in the end of the period
            if step % agg_period == agg_period - 1:
                dic_density[agg_period].append(
                    dic_binary_adjs[agg_period][step].sum()
                    / (n_banks * (n_banks - 1.0))
                )  # for a directed graph
            # otherwise, just extend the time series
            else:
                dic_density[agg_period].append(dic_density[agg_period][-1])

    return dic_density


def get_degree_distribution(dic_binary_adjs):

    # initialise the time series of the jacard index
    dic_in_degree_distribution = {}
    dic_out_degree_distribution = {}
    for agg_period in dic_binary_adjs.keys():
        dic_in_degree_distribution.update({agg_period: [0.0]})
        dic_out_degree_distribution.update({agg_period: [0.0]})

    for step in range(1, len(list(dic_binary_adjs.values())[0])):
        # Build the degree distribution time series - version aggregated.
        for agg_period in dic_binary_adjs.keys():

            # if is in the end of the period
            if step % agg_period == agg_period - 1:
                # first define a networkx object.
                bank_network = nx.from_numpy_matrix(
                    dic_binary_adjs[agg_period][step],
                    parallel_edges=False,
                    create_using=nx.DiGraph,
                )
                # build an array of the in_degree per bank
                ar_in_degree = np.array(bank_network.in_degree())[:, 1]
                ar_out_degree = np.array(bank_network.out_degree())[:, 1]

                dic_in_degree_distribution[agg_period].append(ar_in_degree)
                dic_out_degree_distribution[agg_period].append(ar_out_degree)

            # otherwise, just extend the time series
            else:
                dic_in_degree_distribution[agg_period].append(
                    dic_in_degree_distribution[agg_period][-1]
                )
                dic_out_degree_distribution[agg_period].append(
                    dic_out_degree_distribution[agg_period][-1]
                )

    return dic_in_degree_distribution, dic_out_degree_distribution


def get_binary_adjs(dic_obs_adj_tr, agg_periods):

    n_banks = len(list(dic_obs_adj_tr.values())[0])

    # dictionary of the aggregated ajency matrix over a given period
    dic_binary_adj = {}
    for agg_period in agg_periods:
        dic_binary_adj.update({agg_period: np.zeros((n_banks, n_banks))})

    # dictionary of the time series of the aggregated adjency matrix
    dic_binary_adjs = {}
    for agg_period in agg_periods:
        dic_binary_adjs.update({agg_period: []})

    # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
    for (step, ts_trade) in enumerate(dic_obs_adj_tr.keys()):

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(dic_obs_adj_tr[ts_trade] > 0, True, False)

        for agg_period in agg_periods:

            # build the dic_binary_adj on a given step
            if step % agg_period > 0:
                dic_binary_adj.update(
                    {
                        agg_period: np.logical_or(
                            binary_adj, dic_binary_adj[agg_period]
                        )
                    }
                )
            elif step % agg_period == 0:
                dic_binary_adj.update({agg_period: binary_adj})

            # store the results in a list for each agg period
            dic_binary_adjs[agg_period].append(dic_binary_adj[agg_period])

    return dic_binary_adjs


def get_n_plot_cp_test(dic_binary_adjs, save_every, path_results):

    dic_p_value = {}
    for agg_period in dic_binary_adjs.keys():
        dic_p_value.update({agg_period: [1]})

    for step in np.arange(
        len(list(dic_binary_adjs.values())[0]), step=save_every
    ):

        for agg_period in dic_binary_adjs.keys():

            # build nx object
            bank_network = nx.from_numpy_matrix(
                dic_binary_adjs[agg_period][step],
                parallel_edges=False,
                create_using=nx.DiGraph,
            )

            # run cpnet test
            sig_c, sig_x, significant, p_value = fct.cpnet_test(bank_network)

            # store the p_value
            dic_p_value[agg_period].append(p_value)

            # plot
            fct.init_path(
                path_results + str(agg_period) + "_step_core-periphery/"
            )
            gx.plot_core_periphery(
                bank_network=bank_network,
                sig_c=sig_c,
                sig_x=sig_x,
                path=path_results + str(agg_period) + "_step_core-periphery/",
                step=step,
                name_in_title="reverse repo",
            )

    return dic_p_value
