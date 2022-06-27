from network import InterBankNetwork


if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=10,
        alpha_pareto=1,
        beta_init=80,
        beta_reg=10,
        beta_star=80,
        initial_mr=0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="dirichlet",
        std_law=1.0,
        result_location="./results/",
    )

    network.simulate(100, 5000, 5000)