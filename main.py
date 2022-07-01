from network import ClassNetwork
import sys

if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    network = ClassNetwork(
        n_banks=10,
        alpha_pareto=2.1,
        beta_init=10.0,
        beta_reg=10.0,
        beta_star=10.0,
        alpha=1.0,
        gamma=3.0,
        collateral_value=1.0,
        initialization_method="constant",
        shock_method="dirichlet",
        shocks_vol=0.1,
        result_location="./results/",
    )

    network.simulate(time_steps=300, save_every=1000, jaccard_period=20)
