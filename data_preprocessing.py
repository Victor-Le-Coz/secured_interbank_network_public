import pandas as pd
import pickle
from tqdm import tqdm


def build_from_data(df_mmsr):
    """
    input: mmsr_data: filtered on lend and sell
    """
    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(df_mmsr[["cntp_lei", "report_agent_lei"]].values.ravel("K"))

    # define the list of dates in the mmsr database
    mmsr_trade_dates = sorted(list(set(df_mmsr.index.strftime("%Y-%m-%d"))))

    # initialisation of a dictionary of the observed paths
    dic_obs_adj_cr = {}
    dic_obs_adj_tr = {}
    for mmsr_trade_date in mmsr_trade_dates:
        dic_obs_adj_cr.update(
            {mmsr_trade_date: pd.DataFrame(columns=leis, index=leis, data=0)}
        )
        dic_obs_adj_tr.update(
            {mmsr_trade_date: pd.DataFrame(columns=leis, index=leis, data=0)}
        )

    # building of the matrices and storage in the dictionary observed_path
    for ts_trade in tqdm(df_mmsr.index):

        dic_obs_adj_tr[ts_trade.strftime("%Y-%m-%d")].loc[
            df_mmsr.loc[ts_trade, "report_agent_lei"],
            df_mmsr.loc[ts_trade, "cntp_lei"],
        ] = (
            dic_obs_adj_cr[ts_trade.strftime("%Y-%m-%d")].loc[
                df_mmsr.loc[ts_trade, "report_agent_lei"],
                df_mmsr.loc[ts_trade, "cntp_lei"],
            ]
            + df_mmsr.loc[ts_trade, "trns_nominal_amt"]
        )

        # loop over the dates up to the maturity of the trade
        for date in pd.period_range(
            start=ts_trade,
            end=min(
                df_mmsr.loc[ts_trade, "maturity_time_stamp"],
                pd.to_datetime(mmsr_trade_dates[-1]),
            ),
            freq="1d",
        ).strftime("%Y-%m-%d"):
            dic_obs_adj_cr[date].loc[
                df_mmsr.loc[ts_trade, "report_agent_lei"],
                df_mmsr.loc[ts_trade, "cntp_lei"],
            ] = (
                dic_obs_adj_cr[date].loc[
                    df_mmsr.loc[ts_trade, "report_agent_lei"],
                    df_mmsr.loc[ts_trade, "cntp_lei"],
                ]
                + df_mmsr.loc[ts_trade, "trns_nominal_amt"]
            )

    pickle.dump(
        dic_obs_adj_cr,
        open("./support/dic_obs_adj.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    pickle.dump(
        dic_obs_adj_tr,
        open("./support/dic_obs_adj_tr.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return dic_obs_adj_cr, dic_obs_adj_tr
