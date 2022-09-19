from network import single_run
import sys

# set the recursion limit to an higher value
sys.setrecursionlimit(5000)

single_run(
    n_banks=50,
    alpha=0.01,
    beta_init=0.5,  # for the initial collateral available
    beta_reg=0.5,
    beta_star=0.5,
    gamma=0.03,
    collateral_value=1.0,
    initialization_method="pareto",
    alpha_pareto=1.3,
    shocks_method="dirichlet",
    shocks_law="normal",
    shocks_vol=0.9,
    result_location="./results/single_run/dirichlet/",
    min_repo_size=1e-10,
    time_steps=1000,
    save_every=2500,
    jaccard_periods=[20, 100, 250, 500],
    output_opt=False,
    LCR_mgt_opt=True,
    output_keys=None,
)
