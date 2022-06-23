from network import InterBankNetwork


if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=50,
        alpha_pareto=0.5,
        beta_lcr=50,
        beta_star_lcr=50,
        initial_mr=49,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="uniform",
        std_law=0.30,
        result_location="./results/",
    )

    network.simulate(50, 10, 10)
