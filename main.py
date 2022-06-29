from network import InterBankNetwork


if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=100,
        alpha_pareto=1,
        beta_init=80,
        beta_reg=10,
        beta_star=80,
        initial_mr=0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="conservative",
        std_law=0.1,
        result_location="./results/",
    )

    network.simulate(10000, 10, 50)
