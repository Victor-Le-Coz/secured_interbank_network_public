import function as fct
import sys

# set the recursion limit to an higher value
sys.setrecursionlimit(5000)

fct.single_run(
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
    result_location="./results/single_run/",
    min_repo_size=0.0,
    time_steps=10000,
    save_every=10000,
    jaccard_period=20,
    output_opt=False,
)
