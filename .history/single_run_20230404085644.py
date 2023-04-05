from dynamics import single_run
import sys
import parameters as par

# set the recursion limit to an higher value
sys.setrecursionlimit(5000)

if __name__ == "__main__":
    single_run(
        n_banks=50,
        alpha_init=0.1,  # initial cash (< 1/(1-gamma) - beta)
        alpha=0.01,
        beta_init=1,  # initial collateral  (< 1/(1-gamma) - alpha)
        beta_reg=0.5,
        beta_star=0.5,
        gamma=0.5,
        collateral_value=1.0,
        initialization_method="pareto",
        alpha_pareto=1.2,
        shocks_method="non-conservative",
        shocks_law="normal-mean-reverting",
        shocks_vol=0.01,
        result_location="./results/single_run/general-testing/",
        min_repo_size=1e-8,
        nb_steps=int(1e2),
        save_every=2500,
        jaccard_periods=par.agg_periods,
        agg_periods=par.agg_periods,
        cp_option=True,
        LCR_mgt_opt=False,
        output_keys=False,
    )
