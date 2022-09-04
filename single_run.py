from network import single_run
import sys

# set the recursion limit to an higher value
sys.setrecursionlimit(5000)

single_run(
    n_banks=2,
    alpha=0.01,
    beta_init=0.09,
    beta_reg=0.1,
    beta_star=0.1,
    gamma=0.03,
    collateral_value=1.0,
    initialization_method="constant",
    alpha_pareto=1.3,
    shocks_method="bilateral",
    shocks_law="normal",
    shocks_vol=1,
    result_location="./results/single_run/",
    min_repo_size=1e-10,
    time_steps=1,
    save_every=10000,
    jaccard_period=20,
    output_opt=False,
)
