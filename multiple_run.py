import sys, os
from multiprocessing import Pool
from network import single_run
import functions as fct
import graphics as gx
from socket import gethostname
import metrics as mtr

if __name__ == "__main__":

    # set the recursion limit to an higher value
    sys.setrecursionlimit(5000)

    # define the parameters for the run
    result_location = "./results/general-testing/"
    input_params = [
        "shocks_vol",
        "alpha_pareto",
        "beta",
        "n_banks",
        "min_repo_size",
        "collateral",
    ]
    # axes = [
    #     "collateral",
    # ]
    output_keys = mtr.output_single_keys + mtr.output_mlt_keys
    jaccard_periods = [20, 100, 250, 500]
    agg_periods = [1, 50, 100, 250]

    for input_param in input_params:
        # build the arguments
        args = fct.build_args(
            input_param=input_param,
            n_banks=2,
            alpha=0.01,
            beta_init=0.5,  # for the initial collateral available (must be smaller than 1/(1-gamma) - alpha)
            beta_reg=0.5,
            beta_star=0.5,
            gamma=0.03,
            collateral_value=1.0,
            initialization_method="pareto",
            alpha_pareto=1.3,
            shocks_method="non-conservative",
            shocks_law="normal-mean-reverting",
            shocks_vol=0.01,
            result_location=result_location,
            min_repo_size=1e-8,
            time_steps=2,
            save_every=2500,
            jaccard_periods=jaccard_periods,
            agg_periods=agg_periods,
            cp_option=True,
            output_opt=True,
            LCR_mgt_opt=False,
            output_keys=output_keys,
        )

        # initialize the paths
        fct.init_path(result_location + input_param + "/output_by_args/")

        # run the simulation in multiprocessing across arguments
        if gethostname() == "gibi":
            nb_threads = int(os.cpu_count() / 2)
        else:
            nb_threads = int(os.cpu_count() - 1)
        with Pool(processes=nb_threads) as p:
            output = p.starmap(single_run, args)

        # plot the results
        param_values = fct.get_param_values_testing(input_param)
        gx.plot_output_by_param(
            param_values=param_values,
            input_param=input_param,
            output=output,
            jaccard_periods=jaccard_periods,
            agg_periods=agg_periods,
            path=result_location + input_param + "/output_by_args/",
        )
