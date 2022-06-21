from network import InterBankNetwork


if __name__ == "__main__":
    network = InterBankNetwork(
        n_banks=2,
        alpha_pareto=0.5,
        beta_lcr=10.0,
        beta_star_lcr=10.0,
        initial_mr=1.0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="log-normal",
        std_dirichlet=0.3,
        std_log_normal=0.01,
        result_location="./results/",
    )

network.simulate(1, 1, 30)
