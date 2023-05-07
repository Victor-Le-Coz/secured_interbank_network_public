import dynamics as dyn

if __name__ == "__main__":
    dyn.single_run(
        nb_banks=50,
        alpha_init=0.01,  # initial cash (< 1/(1-gamma) - beta)
        alpha=0.01,
        beta_init=0.5,  # initial collateral  (< 1/(1-gamma) - alpha)
        beta_reg=0.5,
        beta_star=0.5,
        gamma=0.03,  # if too big, the lux version generates huge shocks
        collateral_value=1.0,
        initialization_method="pareto",
        alpha_pareto=1.3,
        shocks_method="non-conservative",
        shocks_law="normal-mean-reverting",
        shocks_vol=0.05,
        result_location="./results/single_run/",
        min_repo_size=1e-8,
        nb_steps=int(2e2),
        dump_period=30,
        plot_period=50,
        cp_option=False,
        LCR_mgt_opt=False,
        output_keys=False,
    )
