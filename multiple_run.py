import sys, os
from multiprocessing import Pool
from network import single_run
import function as fct
import graphics as gx
from socket import gethostname

if __name__ == "__main__":

    # set the recursion limit to an higher value
    sys.setrecursionlimit(5000)

    # define the parameters for the run
    result_location = "./results/"
    axes = [
        "beta",
        "shocks_vol",
        "n_banks",
        "min_repo_size",
        "alpha_pareto",
        "collateral",
    ]
    # axes = [
    #     "n_banks",
    # ]

    for axe in axes:
        # build the arguments
        args = fct.build_args(
            axe=axe,
            n_banks=50,
            alpha=0.01,
            beta_init=0.5,  # for the initial collateral available
            beta_reg=0.5,
            beta_star=0.5,
            gamma=0.03,
            collateral_value=1.0,
            initialization_method="constant",
            alpha_pareto=1.3,
            shocks_method="bilateral",
            shocks_law="normal",
            shocks_vol=0.05,
            result_location=result_location,
            min_repo_size=1e-10,
            time_steps=5000,
            save_every=2500,
            jaccard_period=20,
            output_opt=True,
        )

        # initialize the paths
        fct.init_path(result_location + axe + "/")
        fct.init_path(result_location + axe + "/output_by_args/")
        fct.init_path(result_location + axe + "/output_by_args/log_scale/")

        # run the simulation in multiprocessing across arguments
        if gethostname() == "gibi":
            nb_threads = int(os.cpu_count() / 2)
        else:
            nb_threads = int(os.cpu_count() - 1)
        with Pool(processes=nb_threads) as p:
            output = p.starmap(single_run, args)

        # plot the results
        axe_args = fct.build_axe_args(axe)
        gx.plot_output_by_args(
            axe_args, axe, output, result_location + axe + "/output_by_args/"
        )
