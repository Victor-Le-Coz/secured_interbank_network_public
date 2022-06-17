from network import InterBankNetwork


if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=25,
        alpha_pareto=0.5,
        perc_deposit_shock=0.1,
        beta_lcr=10.0,
        beta_star_lcr=10.0,
        initial_mr=1.0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="normal",
        constant_dirichlet=1,
    )

    network.simulate(10)
