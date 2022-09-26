from network import single_run
import sys

# set the recursion limit to an higher value
sys.setrecursionlimit(5000)

if __name__ == "__main__":
    single_run(
        n_banks=50,
        alpha=0.01,
        beta_init=1.5,  # for the initial collateral available (must be smaller than 1/(1-gamma) - alpha)
        beta_reg=0.5,
        beta_star=0.5,
        gamma=0.5,
        collateral_value=1.0,
        initialization_method="pareto",
        alpha_pareto=1.3,
        shocks_method="non-conservative",
        shocks_law="normal-mean-reverting",
        shocks_vol=0.05,
        result_location="./results/single_run/test/",
        min_repo_size=1e-9,
        time_steps=10000,
        save_every=2500,
        jaccard_periods=[20, 100, 250, 500],
        agg_periods=[50, 100, 200],
        cp_option=True,
        output_opt=False,
        LCR_mgt_opt=False,
        output_keys=None,
    )
