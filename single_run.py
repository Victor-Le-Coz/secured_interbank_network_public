import dynamics as dyn

dyn.single_run(
    nb_banks=150,
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
    min_repo_trans_size=1e-8,
    nb_steps=int(1e4),
    dump_period=2500,
    plot_period=1000,
    cp_option=False,
    LCR_mgt_opt=False,
)
