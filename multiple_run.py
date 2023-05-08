import os
from multiprocessing import Pool
import functions as fct
import parameters as par
import dynamics as dyn
import numpy as np
import dask
from cluster import launch_cluster_mltp

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
    "result_location": "./results/multiple_run/",
    "min_repo_trans_size": 1e-8,
    "nb_steps": int(1e1),
    "dump_period": 10,
    "plot_period": 10,
    "cp_option": False,
    "LCR_mgt_opt": False,
}

dic_ranges = {
    "nb_banks": np.arange(10, 260, 10),
    "alpha_init": np.arange(0, 0.3, 0.01),
    "beta_init": np.arange(0, 1, 0.05),
    "beta_reg": np.arange(0.01, 1, 0.02),
    "alpha_pareto": np.logspace(0, 1, num=25),
    "shocks_vol": np.arange(0, 0.30, 0.0025),
    "min_repo_trans_size": np.logspace(-16, 2, num=25),
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
    list_dic_args = fct.build_args(dic_default_value, dic_ranges)

    # run the dask cluster o
    client, cluster = launch_cluster_mltp(
        TASK_MEMORY=19,
        JOB_WALLTIME="25:00:00",
    )

    dld_obj = []
    for dic_args in list_dic_args:
        dld_obj.append(dask.delayed(dyn.single_run)(**dic_args))

    dask.compute(dld_obj)

    # # plot the results # to be re-factored
    # param_values = fct.get_param_values(input_param)
    # gx.plot_output_by_param(
    #     param_values=param_values,
    #     input_param=input_param,
    #     output=output,
    #     jaccard_periods=jaccard_periods,
    #     agg_periods=agg_periods,
    #     path=result_location + input_param + "/output_by_args/",
    # )
