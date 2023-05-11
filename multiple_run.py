import functions as fct
import dynamics as dyn
import numpy as np
import dask
from cluster import launch_cluster_mltp
import graphics as gx

path = "./results/testing/sensitivity/"

dic_default_value = {
    "nb_banks": 50,
    "alpha_init": 0.01,  # initial cash (< 1/(1-gamma) - beta)
    "alpha": 0.01,
    "beta_init": 0.5,  # initial collateral  (< 1/(1-gamma) - alpha)
    "beta_reg": 0.5,
    "beta_star": 0.5,
    "gamma": 0.03,  # if too big, the lux version generates huge shocks
    "collateral_value": 1.0,
    "initialization_method": "pareto",
    "alpha_pareto": 1.3,
    "shocks_method": "non-conservative",
    "shocks_law": "normal-mean-reverting",
    "shocks_vol": 0.05,
    "result_location": f"{path}runs/",
    "min_repo_trans_size": 1e-8,
    "nb_steps": int(5e3),
    "dump_period": int(5e3),
    "plot_period": int(5e2),
    "cp_option": True,
    "LCR_mgt_opt": False,
}

dic_range = {
    "nb_banks": np.arange(10, 260, 5),
    "alpha_init": np.arange(0, 0.3, 0.01),
    "beta_init": np.arange(0, 1, 0.02),
    "beta_reg": np.arange(0.01, 1, 0.01),
    "alpha_pareto": np.logspace(0, 1, num=50),
    "shocks_vol": np.arange(0, 0.30, 0.0010),
    "min_repo_trans_size": np.logspace(-16, 2, num=50),
}


dic_ranges_test = {
    "nb_banks": np.arange(1, 3),
    "alpha_init": np.arange(0, 1, 0.1),
    "beta_init": np.arange(0, 1, 0.05),
    "beta_reg": np.arange(0.01, 0.03, 0.01),
    "alpha_pareto": np.logspace(0, 1, num=3),
    "shocks_vol": np.arange(0, 0.30, 0.0025),
    "min_repo_trans_size": np.logspace(-16, 2, num=3),
}


if __name__ == "__main__":

    # build list of the dic_args to be tested
    list_dic_args = fct.build_args(dic_default_value, dic_range)

    # open a cluster
    client, cluster = launch_cluster_mltp(
        TASK_MEMORY=19,
        JOB_WALLTIME="30:00:00",
        # max_cpu=100,
    )

    # run with dask distributed
    dld_obj = [
        dask.delayed(dyn.single_run)(**dic_args) for dic_args in list_dic_args
    ]
    dask.compute(dld_obj)

    # collect results into df_network_sensitivity
    df_network_sensitivity = fct.get_df_network_sensitivity(
        dic_default_value["result_location"]
    )

    # plot the sensitivity
    gx.plot_all_sensitivities(df_network_sensitivity, path=path)
