import sys
from multiprocessing import Pool
import function as fct
import graphics as gx

if __name__ == "__main__":

    # set the recursion limit to an higher value
    sys.setrecursionlimit(5000)

    # define the parameters for the run
    result_location = "./results/"
    axes = ["n_banks", "beta", "shocks_vol", "min_repo_size"]
    # axes = ["min_repo_size"]

    for axe in axes:
        # build the arguments
        args = fct.build_args(
            axe=axe,
            n_banks=50,
            alpha_pareto=1.3,
            beta_init=0.1,
            beta_reg=0.1,
            beta_star=0.1,
            alpha=0.01,
            gamma=0.03,
            collateral_value=1.0,
            initialization_method="constant",
            shock_method="bilateral",
            shocks_vol=0.05,
            result_location=result_location,
            min_repo_size=0.0,
            time_steps=5000,
            save_every=2500,
            jaccard_period=20,
            output_opt=True,
        )

        # initialize the paths
        fct.init_path(result_location + axe + "/")
        fct.init_path(result_location + axe + "/output_by_args/")

        # run the simulation in multiprocessing across arguments
        with Pool(processes=8) as p:
            output = p.starmap(fct.single_run, args)

        # plot the results
        axe_args = fct.build_axe_args(axe)
        gx.plot_output_by_args(
            axe_args, axe, output, result_location + axe + "/output_by_args/"
        )
