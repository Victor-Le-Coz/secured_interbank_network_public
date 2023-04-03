import sys, os
from multiprocessing import Pool
from network import single_run
import functions as fct
import graphics as gx
from socket import gethostname
import parameters as mtr

if __name__ == "__main__":

    # set the recursion limit to an higher value
    sys.setrecursionlimit(5000)

    # define the parameters for the run
    result_location = "./results/general-testing/"
    input_params = mtr.input_params
    output_keys = mtr.output_single_keys + mtr.output_mlt_keys
    jaccard_periods = [20, 100, 250, 500]
    agg_periods = [1, 50, 100, 250]

    for input_param in input_params:
        # build the arguments
        args = fct.build_args(
            input_param=input_param,
            n_banks=50,
            alpha_init=0.1,  # initial cash (< 1/(1-gamma) - beta)
            alpha=0.01,
            beta_init=1,  # initial collateral  (< 1/(1-gamma) - alpha)
            beta_reg=0.5,
            beta_star=0.5,
            gamma=0.5,
            collateral_value=1.0,
            initialization_method="pareto",
            alpha_pareto=1.3,
            shocks_method="non-conservative",
            shocks_law="normal-mean-reverting",
            shocks_vol=0.01,
            result_location=result_location,
            min_repo_size=1e-8,
            time_steps=int(1e4),
            save_every=2500,
            jaccard_periods=jaccard_periods,
            agg_periods=agg_periods,
            cp_option=True,
            output_opt=True,
            LCR_mgt_opt=False,
            output_keys=output_keys,
        )

        # initialize the paths
        fct.delete_n_init_path(
            result_location + input_param + "/output_by_args/"
        )

        # run the simulation in multiprocessing across arguments
        if gethostname() == "gibi":
            nb_threads = int(os.cpu_count() / 2)
        elif gethostname() == "srv006542.fr.cfm.fr":
            nb_threads = int(os.cpu_count() * 3 / 4)
        else:
            nb_threads = int(os.cpu_count())
        with Pool(processes=nb_threads) as p:
            output = p.starmap(single_run, args)

        # plot the results
        param_values = fct.get_param_values(input_param)
        gx.plot_output_by_param(
            param_values=param_values,
            input_param=input_param,
            output=output,
            jaccard_periods=jaccard_periods,
            agg_periods=agg_periods,
            path=result_location + input_param + "/output_by_args/",
        )
