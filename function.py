import numpy as np
import cpnet  # Librairy for the estimation of core-periphery structures
import os
import shutil
from network import ClassNetwork


def gini(x):
    """
    This function computes the gini coeficient of a numpy arary.
    param: x: a numpy array
    return: the gini coeficient
    """
    total = 0
    for i, xi in enumerate(x):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def build_axe_args(axe):
    if axe == "n_banks":
        axe_args = [n_banks_test for n_banks_test in np.arange(10, 50, 10)]
    elif axe == "beta":
        axe_args = [beta for beta in np.arange(0.02, 0.18, 0.02)]
    elif axe == "shocks_vol":
        axe_args = [shocks_vol_test for shocks_vol_test in np.arange(0.02, 0.18, 0.02)]
    elif axe == "min_repo_size":
        axe_args = [
            min_repo_size_test for min_repo_size_test in np.logspace(-16, -8, num=8)
        ]
    return axe_args


def build_args(
    axe,
    n_banks=50,
    alpha_pareto=1.3,
    beta_init=0.1,
    beta_reg=0.1,
    beta_star=0.1,
    alpha=0.01,
    gamma=0.03,
    collateral_value=1.0,
    initialization_method="constant",
    shock_method="bilateral",
    shocks_vol=0.05,
    result_location="./results/",
    min_repo_size=0.0,
    time_steps=500,
    save_every=500,
    jaccard_period=20,
    output_opt=False,
):

    args = []

    axe_args = build_axe_args(axe)

    if axe == "n_banks":
        for axe_arg in axe_args:
            args.append(
                (
                    axe_arg,
                    alpha_pareto,
                    beta_init,
                    beta_reg,
                    beta_star,
                    alpha,
                    gamma,
                    collateral_value,
                    initialization_method,
                    shock_method,
                    shocks_vol,
                    result_location + axe + "/" + str(axe_arg) + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_period,
                    output_opt,
                )
            )

    elif axe == "beta":
        for axe_arg in axe_args:
            args.append(
                (
                    n_banks,
                    alpha_pareto,
                    axe_arg,
                    axe_arg,
                    axe_arg,
                    alpha,
                    gamma,
                    collateral_value,
                    initialization_method,
                    shock_method,
                    shocks_vol,
                    result_location + axe + "/" + str(axe_arg) + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_period,
                    output_opt,
                )
            )

    elif axe == "shocks_vol":
        for axe_arg in axe_args:
            args.append(
                (
                    n_banks,
                    alpha_pareto,
                    beta_init,
                    beta_reg,
                    beta_star,
                    alpha,
                    gamma,
                    collateral_value,
                    initialization_method,
                    shock_method,
                    axe_arg,
                    result_location + axe + "/" + str(axe_arg) + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_period,
                    output_opt,
                )
            )

    elif axe == "min_repo_size":
        for axe_arg in axe_args:
            args.append(
                (
                    n_banks,
                    alpha_pareto,
                    beta_init,
                    beta_reg,
                    beta_star,
                    alpha,
                    gamma,
                    collateral_value,
                    initialization_method,
                    shock_method,
                    shocks_vol,
                    result_location + axe + "/" + str(axe_arg) + "/",
                    axe_arg,
                    time_steps,
                    save_every,
                    jaccard_period,
                    output_opt,
                )
            )

    return args


def reformat_output(output):
    """
    Function to convert a list of dictionaries into a dictionary of lists.
    """
    if type(output) is dict:  # case of single run
        return output

    else:  # case of multiprocessing
        # initialization with empty list for each keys
        output_rf = {}
        for keys in output[0].keys():
            output_rf.update({keys: []})

        # build the lists within the dict
        for output_dict in output:
            for keys in output_dict.keys():
                output_rf[keys].append(output_dict[keys])
        return output_rf


def cpnet_test(
    bank_network,
):
    alg = cpnet.BE()  # Load the Borgatti-Everett algorithm
    alg.detect(bank_network)  # Feed the network as an input
    x = alg.get_coreness()  # Get the coreness of nodes
    c = alg.get_pair_id()  # Get the group membership of nodes

    # Statistical significance test
    sig_c, sig_x, significant, p_value = cpnet.qstest(
        c,
        x,
        bank_network,
        alg,
        significance_level=0.05,
        num_of_thread=4,
    )

    print(
        "{} core-periphery structure(s) detected, but {} significant, "
        "p-values are {} "
        "".format(len(significant), np.sum(significant), p_value)
    )

    return sig_c, sig_x, significant, p_value


def init_path(path):
    if os.path.exists(path):  # Delete all previous figures
        shutil.rmtree(path)
    os.makedirs(path)  # create the path


def single_run(
    n_banks=10,
    alpha_pareto=1.3,
    beta_init=0.1,
    beta_reg=0.1,
    beta_star=0.1,
    alpha=0.01,
    gamma=0.03,
    collateral_value=1.0,
    initialization_method="constant",
    shock_method="bilateral",
    shocks_vol=0.05,
    result_location="./results/",
    min_repo_size=0.0,
    time_steps=500,
    save_every=500,
    jaccard_period=20,
    output_opt=False,
):

    network = ClassNetwork(
        n_banks=n_banks,
        alpha_pareto=alpha_pareto,
        beta_init=beta_init,
        beta_reg=beta_reg,
        beta_star=beta_star,
        alpha=alpha,
        gamma=gamma,
        collateral_value=collateral_value,
        initialization_method=initialization_method,
        shock_method=shock_method,
        shocks_vol=shocks_vol,
        result_location=result_location,
        min_repo_size=min_repo_size,
    )

    if output_opt:
        return network.simulate(
            time_steps=time_steps,
            save_every=save_every,
            jaccard_period=jaccard_period,
            output_opt=output_opt,
        )

    else:
        network.simulate(
            time_steps=time_steps,
            save_every=save_every,
            jaccard_period=jaccard_period,
            output_opt=output_opt,
        )
