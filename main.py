from network import ClassNetwork
import sys

if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    network = ClassNetwork(
        n_banks=50,
        alpha_pareto=3,
        beta_init=0.1,
        beta_reg=0.1,
        beta_star=0.1,
        alpha=0.01,
        gamma=0.03,
        collateral_value=1.0,
        initialization_method="pareto",
        shock_method="dirichlet",
        shocks_vol=0.1,
        result_location="./results/",
    )

    network.simulate(time_steps=300, save_every=300, jaccard_period=20)
