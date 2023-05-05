import sys
import parameters as par
from network import ClassNetwork
from dynamics import ClassDynamics

# set the recursion limit to an higher value
sys.setrecursionlimit(5000)


def single_run(
    nb_banks,
    alpha_init,
    alpha,
    beta_init,
    beta_reg,
    beta_star,
    gamma,
    collateral_value,
    initialization_method,
    alpha_pareto,
    shocks_method,
    shocks_law,
    shocks_vol,
    result_location,
    min_repo_size,
    nb_steps,
    save_every,
    jaccard_periods,
    agg_periods,
    cp_option,
    LCR_mgt_opt,
    output_keys,
):

    Network = ClassNetwork(
        nb_banks=nb_banks,
        alpha_init=alpha_init,
        beta_init=beta_init,
        beta_reg=beta_reg,
        beta_star=beta_star,
        alpha=alpha,
        gamma=gamma,
        collateral_value=collateral_value,
        initialization_method=initialization_method,
        alpha_pareto=alpha_pareto,
        shocks_method=shocks_method,
        shocks_law=shocks_law,
        shocks_vol=shocks_vol,
        min_repo_size=min_repo_size,
        LCR_mgt_opt=LCR_mgt_opt,
    )

    dynamics = ClassDynamics(
        Network,
        nb_steps=nb_steps,
        path_results=result_location,
        jaccard_periods=jaccard_periods,
        agg_periods=agg_periods,
        cp_option=cp_option,
    )

    output = dynamics.simulate(
        save_every=save_every,
        output_keys=output_keys,
    )

    return output


if __name__ == "__main__":
    single_run(
        nb_banks=50,
        alpha_init=0.01,  # initial cash (< 1/(1-gamma) - beta)
        alpha=0.01,
        beta_init=0.5,  # initial collateral  (< 1/(1-gamma) - alpha)
        beta_reg=0.5,
        beta_star=0.5,
        gamma=0.03,  # if it is too big, the lux version generates huge shocks, it kills the network as the central bank funding explodes
        collateral_value=1.0,
        initialization_method="pareto",
        alpha_pareto=1.3,
        shocks_method="non-conservative",
        shocks_law="normal-mean-reverting",
        shocks_vol=0.05,
        result_location="./results/single_run/",
        min_repo_size=1e-8,
        nb_steps=int(1e4),
        save_every=2500,
        jaccard_periods=par.agg_periods,
        agg_periods=par.agg_periods,
        cp_option=True,
        LCR_mgt_opt=False,
        output_keys=False,
    )
