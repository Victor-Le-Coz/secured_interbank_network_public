from network import ClassNetwork


if __name__ == "__main__":
    network = ClassNetwork(
        n_banks=50,
        alpha_pareto=1,
        beta_init=10.0,
        beta_reg=10.0,
        beta_star=10.0,
        initial_mr=1,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="dirichlet",
        std_law=1.0,
        result_location="./results/",
    )

    network.simulate(time_steps=100, save_every=5000, jaccard_period=20)
