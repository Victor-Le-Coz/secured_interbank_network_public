import pandas as pd
import pickle
from tqdm import tqdm


def build_from_mmsr(df_mmsr):
    """
    input: mmsr_data: filtered on lend and sell
    """
    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(
        df_mmsr[["cntp_lei", "report_agent_lei"]].values.ravel("K")
    )

    # define the list of dates in the mmsr database
    mmsr_trade_dates = sorted(list(set(df_mmsr.index.strftime("%Y-%m-%d"))))

    # initialisation of a dictionary of the observed paths
    dic_obs_matrix_reverse_repo = {}  # for the exposures
    for mmsr_trade_date in mmsr_trade_dates:
        dic_obs_matrix_reverse_repo.update(
            {mmsr_trade_date: pd.DataFrame(columns=leis, index=leis, data=0)}
        )

    # building of the matrices and storage in the dictionary observed_path - as the maturity of the evergreen is one day when it is repeated (and notice perdio when it is closed) one can apply the same rule everywhere
    for ts_trade in tqdm(df_mmsr.index):
        if df_mmsr.loc[ts_trade, "trns_type"] in ["LEND", "SELL"]:
            for date in pd.period_range(
                start=ts_trade,
                end=min(
                    df_mmsr.loc[ts_trade, "maturity_time_stamp"],
                    pd.to_datetime(mmsr_trade_dates[-1]),
                ),
                freq="1d",
            ).strftime("%Y-%m-%d"):
                dic_obs_matrix_reverse_repo[date].loc[
                    df_mmsr.loc[ts_trade, "cntp_lei"],
                    df_mmsr.loc[ts_trade, "report_agent_lei"],
                ] = (
                    dic_obs_matrix_reverse_repo[date].loc[
                        df_mmsr.loc[ts_trade, "cntp_lei"],
                        df_mmsr.loc[ts_trade, "report_agent_lei"],
                    ]
                    + df_mmsr.loc[ts_trade, "trns_nominal_amt"]
                )

    pickle.dump(
        dic_obs_matrix_reverse_repo,
        open("./support/dic_obs_matrix_reverse_repo.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return dic_obs_matrix_reverse_repo


def get_dic_obs_adj_cr():
    return pickle.load(
        open("./support/dic_obs_matrix_reverse_repo.pickle", "rb")
    )


def build_from_exposures(df_exposures):

    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(df_exposures[["borr_lei", "lend_lei"]].values.ravel("K"))

    # define the list of dates in the mmsr database
    # mmsr_trade_dates = pd.unique(df_exposures["Setdate"])
    mmsr_trade_dates = sorted(list(set(df_exposures["Setdate"])))

    # initialisation of a dictionary of the observed paths
    dic_obs_matrix_reverse_repo = {}  # for the exposures
    for mmsr_trade_date in mmsr_trade_dates:
        dic_obs_matrix_reverse_repo.update(
            {mmsr_trade_date: pd.DataFrame(columns=leis, index=leis, data=0)}
        )

    # building of the matrices and storage in the dictionary observed_path - as the maturity of the evergreen is one day when it is repeated (and notice perdio when it is closed) one can apply the same rule everywhere
    for index in tqdm(df_exposures.index):
        ts_trade = df_exposures.loc[index, "Setdate"]
        dic_obs_matrix_reverse_repo[ts_trade].loc[
            df_exposures.loc[index, "lend_lei"],
            df_exposures.loc[index, "borr_lei"],
        ] = df_exposures.loc[index, "exposure"]

    pickle.dump(
        dic_obs_matrix_reverse_repo,
        open("./support/dic_obs_matrix_reverse_repo.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return dic_obs_matrix_reverse_repo
