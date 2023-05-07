import pandas as pd
import pickle
from tqdm import tqdm
import functions as fct
import numpy as np
from numba import jit
import parameters as par


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

    fct.init_path("./support")
    pickle.dump(
        dic_obs_matrix_reverse_repo,
        open("./support/dic_obs_matrix_reverse_repo.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return dic_obs_matrix_reverse_repo


def load_dic_obs_matrix_reverse_repo():
    return pickle.load(
        open("./support/dic_obs_matrix_reverse_repo.pickle", "rb")
    )


def build_from_exposures(df_exposures, path):

    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(df_exposures[["borr_lei", "lend_lei"]].values.ravel("K"))

    # define the list of dates in the mmsr database
    # mmsr_trade_dates = pd.unique(df_exposures["Setdate"])
    mmsr_trade_dates = pd.to_datetime(
        sorted(list(set(df_exposures["Setdate"].dt.strftime("%Y-%m-%d"))))
    )

    # initialisation of a dictionary of the observed paths
    dic_obs_matrix_reverse_repo = {}  # for the exposures
    for mmsr_trade_date in mmsr_trade_dates:
        dic_obs_matrix_reverse_repo.update(
            {mmsr_trade_date: pd.DataFrame(columns=leis, index=leis, data=0)}
        )

    # building of the matrices and storage in the dictionary observed_path - as the maturity of the evergreen is one day when it is repeated (and notice perdio when it is closed) one can apply the same rule everywhere
    for index in tqdm(df_exposures.index):
        ts_trade = pd.to_datetime(
            df_exposures.loc[index, "Setdate"].strftime("%Y-%m-%d")
        )
        dic_obs_matrix_reverse_repo[ts_trade].loc[
            df_exposures.loc[index, "lend_lei"],
            df_exposures.loc[index, "borr_lei"],
        ] = df_exposures.loc[index, "exposure"]

    fct.init_path("./support")
    pickle.dump(
        dic_obs_matrix_reverse_repo,
        open(f"{path}dic_obs_matrix_reverse_repo.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return dic_obs_matrix_reverse_repo


@jit(nopython=True)
def fast_build_arr_binary_adj(
    arr_obs_matrix_reverse_repo, arr_agg_period, nb_days
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
                    arr_obs_matrix_reverse_repo[agg_day_nb] > 0, True, False
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


def build_rolling_binary_adj(dic_obs_matrix_reverse_repo, path):

    # convert dic to array
    bank_ids = list(list(dic_obs_matrix_reverse_repo.values())[0].index)
    nb_banks = len(bank_ids)
    days = list(dic_obs_matrix_reverse_repo.keys())
    arr_obs_matrix_reverse_repo = np.fromiter(
        dic_obs_matrix_reverse_repo.values(),
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

    # dump results
    pickle.dump(
        dic_arr_binary_adj,
        open(f"{path}dic_arr_binary_adj.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return dic_arr_binary_adj


def load_dic_arr_binary_adj():
    return pickle.load(open("./support/dic_arr_binary_adj.pickle", "rb"))
