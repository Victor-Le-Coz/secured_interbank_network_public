from network import ClassNetwork
import sys

if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    network = ClassNetwork(
        n_banks=50,
        alpha_pareto=2.1,
        beta_init=10.0,
        beta_reg=10.0,
        beta_star=20.0,
        initial_mr=1,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="pareto",
        shock_method="dirichlet",
        shocks_vol=1,
        result_location="./results/",
    )

    network.simulate(time_steps=1000, save_every=1000, jaccard_period=20)

