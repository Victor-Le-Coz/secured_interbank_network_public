import numpy as np

# jaccard index of the transactions: better approach to mesure the stability of trading relationships when maturities are longer than one day


def compute_jaccard(dic_obs_adj_tr, agg_periods):

    n_banks = len(list(dic_obs_adj_tr.values())[0])

    # initialise the time series of the jacard index
    dic_jaccard = {}
    for agg_period in agg_periods:
        dic_jaccard.update({str(agg_period) + " time steps": []})

    # dictionary of the aggregated ajency matrix over a given period
    agg_binary_adj_dic = {}
    for agg_period in agg_periods:
        agg_binary_adj_dic.update({agg_period: np.zeros((n_banks, n_banks))})

    # dictionary of the previous aggregated ajency matrix over a given period
    prev_agg_binary_adj_dic = {}
    for agg_period in agg_periods:
        prev_agg_binary_adj_dic.update({agg_period: np.zeros((n_banks, n_banks))})

    # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
    for (step, ts_trade) in enumerate(dic_obs_adj_tr.keys()):

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(dic_obs_adj_tr[ts_trade] > 0, True, False)

        if step > 0:
            for agg_period in agg_periods:
                if step % agg_period > 0:
                    agg_binary_adj_dic.update(
                        {
                            agg_period: np.logical_or(
                                binary_adj, agg_binary_adj_dic[agg_period]
                            )
                        }
                    )
                elif step % agg_period == 0:
                    agg_binary_adj_dic.update({agg_period: binary_adj})

        # Build the jaccard index time series - version aggregated.
        for key in dic_jaccard.keys():
            dic_jaccard[key].append(0.0)

        for agg_period in agg_periods:
            if step % agg_period == agg_period - 1:
                dic_jaccard[str(agg_period) + " time steps"][-1] = (
                    np.logical_and(
                        agg_binary_adj_dic[agg_period],
                        prev_agg_binary_adj_dic[agg_period],
                    ).sum()
                    / np.logical_or(
                        agg_binary_adj_dic[agg_period],
                        prev_agg_binary_adj_dic[agg_period],
                    ).sum()
                )
                prev_agg_binary_adj_dic.update(
                    {agg_period: agg_binary_adj_dic[agg_period].copy()}
                )
            elif step > 0:
                dic_jaccard[str(agg_period) + " time steps"][-1] = dic_jaccard[
                    str(agg_period) + " time steps"
                ][-2]

    return dic_jaccard


def compute_density(dic_obs_adj_tr, agg_periods):

    n_banks = len(list(dic_obs_adj_tr.values())[0])

    # initialise the time series of the jacard index
    dic_density = {}
    for agg_period in agg_periods:
        dic_density.update({str(agg_period) + " time steps": []})

    # dictionary of the aggregated ajency matrix over a given period
    agg_binary_adj_dic = {}
    for agg_period in agg_periods:
        agg_binary_adj_dic.update({agg_period: np.zeros((n_banks, n_banks))})

    # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
    for (step, ts_trade) in enumerate(dic_obs_adj_tr.keys()):

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(dic_obs_adj_tr[ts_trade] > 0, True, False)

        if step > 0:
            for agg_period in agg_periods:
                if step % agg_period > 0:
                    agg_binary_adj_dic.update(
                        {
                            agg_period: np.logical_or(
                                binary_adj, agg_binary_adj_dic[agg_period]
                            )
                        }
                    )
                elif step % agg_period == 0:
                    agg_binary_adj_dic.update({agg_period: binary_adj})

        # Build the jaccard index time series - version aggregated.
        for key in dic_density.keys():
            dic_density[key].append(0.0)

        for agg_period in agg_periods:
            if step % agg_period == agg_period - 1:
                dic_density[str(agg_period) + " time steps"][-1] = agg_binary_adj_dic[
                    agg_period
                ].sum() / (
                    n_banks * (n_banks - 1.0)
                )  # for a directed graph
            elif step > 0:
                dic_density[str(agg_period) + " time steps"][-1] = dic_density[
                    str(agg_period) + " time steps"
                ][-2]

    return dic_density


def get_binary_adj(dic_obs_adj_tr, agg_periods):

    n_banks = len(list(dic_obs_adj_tr.values())[0])

    # dictionary of the aggregated ajency matrix over a given period
    dic_binary_adj = {}
    for agg_period in agg_periods:
        dic_binary_adj.update({agg_period: np.zeros((n_banks, n_banks))})

    # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
    for (step, ts_trade) in enumerate(dic_obs_adj_tr.keys()):

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(dic_obs_adj_tr[ts_trade] > 0, True, False)

        if step > 0:
            for agg_period in agg_periods:
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

    return dic_binary_adj
