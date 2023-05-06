import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import cpnet  # Librairy for the estimation of core-periphery structures
import os
import shutil
import pandas as pd
from scipy import stats
import sys


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


def get_param_values(input_param):
    if input_param == "n_banks":
        param_values = [
            n_banks_test for n_banks_test in np.arange(10, 260, 10)
        ]
    elif input_param == "beta":
        param_values = [beta for beta in np.arange(0.01, 1, 0.02)]
    elif (
        input_param == "collateral"
    ):  # can not be higher than the targeted LCR - except in the no LCR mngt version (lux's model)
        param_values = [
            beta_init_test for beta_init_test in np.arange(0, 1, 0.05)
        ]
    elif input_param == "shocks_vol":
        param_values = [
            shocks_vol_test for shocks_vol_test in np.arange(0, 0.30, 0.0025)
        ]
    elif input_param == "min_repo_size":
        param_values = [
            min_repo_size_test
            for min_repo_size_test in np.logspace(-16, 2, num=25)
        ]
    elif input_param == "alpha_pareto":
        param_values = [
            alpha_pareto_test
            for alpha_pareto_test in np.logspace(0, 1, num=25)
        ]
    elif input_param == "cash":
        param_values = [
            alpha_init_test for alpha_init_test in np.arange(0, 0.3, 0.01)
        ]
    return param_values


def get_param_values_testing(input_param):
    if input_param == "n_banks":
        param_values = [n_banks_test for n_banks_test in np.arange(1, 3)]
    elif input_param == "beta":
        param_values = [beta for beta in np.arange(0.01, 0.03, 0.01)]
    elif (
        input_param == "collateral"
    ):  # can not be higher than the targeted LCR - except in the no LCR mngt version (lux's model)
        param_values = [
            beta_init_test for beta_init_test in np.arange(0, 1, 0.05)
        ]
    elif input_param == "shocks_vol":
        param_values = [
            shocks_vol_test for shocks_vol_test in np.arange(0, 0.30, 0.0025)
        ]
    elif input_param == "min_repo_size":
        param_values = [
            min_repo_size_test
            for min_repo_size_test in np.logspace(-16, 2, num=3)
        ]
    elif input_param == "alpha_pareto":
        param_values = [
            alpha_pareto_test for alpha_pareto_test in np.logspace(0, 1, num=3)
        ]
    elif input_param == "cash":
        param_values = [
            alpha_init_test for alpha_init_test in np.arange(0, 1, 0.1)
        ]

    return param_values


def build_args(
    input_param,
    n_banks=50,
    alpha_init=0.01,
    alpha=0.01,
    beta_init=0.1,
    beta_reg=0.1,
    beta_star=0.1,
    gamma=0.03,
    collateral_value=1.0,
    initialization_method="pareto",
    alpha_pareto=1.3,
    shocks_method="bilateral",
    shocks_law="normal",
    shocks_vol=0.05,
    result_location="./results/",
    min_repo_size=1e-10,
    time_steps=500,
    save_every=500,
    jaccard_periods=[20, 100, 250, 500],
    agg_periods=[20, 100, 250],
    cp_option=False,
    LCR_mgt_opt=True,
    output_keys=None,
):

    args = []

    param_values = get_param_values(input_param)

    if input_param == "n_banks":
        for input_param_value in param_values:
            args.append(
                (
                    input_param_value,
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
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
                )
            )

    elif input_param == "beta":
        for input_param_value in param_values:
            args.append(
                (
                    n_banks,
                    alpha_init,
                    alpha,
                    input_param_value,
                    input_param_value,
                    input_param_value,
                    gamma,
                    collateral_value,
                    initialization_method,
                    alpha_pareto,
                    shocks_method,
                    shocks_law,
                    shocks_vol,
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
                )
            )

    elif input_param == "collateral":
        for input_param_value in param_values:
            args.append(
                (
                    n_banks,
                    alpha_init,
                    alpha,
                    input_param_value,
                    beta_reg,
                    beta_star,
                    gamma,
                    collateral_value,
                    initialization_method,
                    alpha_pareto,
                    shocks_method,
                    shocks_law,
                    shocks_vol,
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
                )
            )

    elif input_param == "shocks_vol":
        for input_param_value in param_values:
            args.append(
                (
                    n_banks,
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
                    input_param_value,
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
                )
            )

    elif input_param == "min_repo_size":
        for input_param_value in param_values:
            args.append(
                (
                    n_banks,
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
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    input_param_value,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
                )
            )

    elif input_param == "alpha_pareto":
        for input_param_value in param_values:
            args.append(
                (
                    n_banks,
                    alpha_init,
                    alpha,
                    beta_init,
                    beta_reg,
                    beta_star,
                    gamma,
                    collateral_value,
                    "pareto",
                    input_param_value,
                    shocks_method,
                    shocks_law,
                    shocks_vol,
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
                )
            )

    elif input_param == "cash":
        for input_param_value in param_values:
            args.append(
                (
                    n_banks,
                    input_param_value,
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
                    result_location
                    + input_param
                    + "/"
                    + str(input_param_value)
                    + "/",
                    min_repo_size,
                    time_steps,
                    save_every,
                    jaccard_periods,
                    agg_periods,
                    cp_option,
                    LCR_mgt_opt,
                    output_keys,
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


def cpnet_test(bank_network, algo="BE"):
    if algo == "KM_ER":  # divide by zero error
        alg = cpnet.KM_ER()
    elif algo == "KM_config":  # divide by zero error
        alg = cpnet.KM_config()
    elif algo == "Divisive":  # divide by zero error
        alg = cpnet.Divisive()
    elif algo == "Rombach":
        alg = cpnet.Rombach()
    elif algo == "Rossa":
        alg = cpnet.Rossa()
    elif algo == "LapCore":
        alg = cpnet.LapCore()
    elif algo == "LapSgnCore":
        alg = cpnet.LapSgnCore()
    elif algo == "LowRankCore":
        alg = cpnet.LowRankCore()
    elif algo == "MINRES":  # do not take weights into acount
        alg = cpnet.MINRES()
    elif algo == "Surprise":  # do not take weights into acount
        alg = cpnet.Surprise()
    elif algo == "Lip":  # do not take weights into acount
        alg = cpnet.Lip()
    elif algo == "BE":  # do not take weights into acount
        alg = cpnet.BE()

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
        num_of_thread=1,
    )

    # print(
    #     "{} core-periphery structure(s) detected, but {} significant, "
    #     "p-values are {} "
    #     "".format(len(significant), np.sum(significant), p_value)
    # )

    return sig_c, sig_x, significant, p_value


def init_results_path(path):
    if os.path.exists(path):  # Delete all previous figures
        shutil.rmtree(path)
    os.makedirs(os.path.join(path, "repo_networks"))
    os.makedirs(os.path.join(path, "trust_networks"))
    os.makedirs(os.path.join(path, "core-periphery_structure"))
    os.makedirs(os.path.join(path, "deposits"))
    os.makedirs(os.path.join(path, "balance_Sheets"))


def delete_n_init_path(path):
    if os.path.exists(path):  # Delete all previous figures
        shutil.rmtree(path)
    os.makedirs(path)


def init_path(path):
    if not (os.path.exists(path)):
        os.makedirs(path)


def dump_np_array(array, name):
    df = pd.DataFrame(array)
    df.to_csv(name)


def get_trunc_lognorm(mu, sigma, lower_bound, upper_bound=np.inf, size=10000):
    norm_lower = np.log(lower_bound)
    norm_upper = np.log(upper_bound)
    X = stats.truncnorm(
        (norm_lower - mu) / sigma,
        (norm_upper - mu) / sigma,
        loc=mu,
        scale=sigma,
    )
    norm_data = X.rvs(size)
    log_norm_data = np.exp(norm_data)
    return log_norm_data


def last_common_element(list1, list2):
    """
    This function returns the last common element between two lists.
    If the lists do not have any common elements, the function returns None.
    """
    # Traverse both lists in reverse order
    for i in range(len(list1) - 1, -1, -1):
        for j in range(len(list2) - 1, -1, -1):
            # If the current element of both lists is equal
            if list1[i] == list2[j]:
                # We have found the common element, return it
                return list1[i]
    # If we get here, there is no common element
    return None


def get_size(obj, seen=None):
    """Recursively finds size of objects. Needs: import sys"""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0  # to handle self-referential objects
    seen.add(id(obj))
    size = sys.getsizeof(obj, 0)  # pypy3 always returns default (necessary)
    if isinstance(obj, dict):
        size += sum(
            get_size(v, seen) + get_size(k, seen) for k, v in obj.items()
        )
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__slots__"):  # in case slots are in use
        slotList = [getattr(C, "__slots__", []) for C in obj.__class__.__mro__]
        slotList = [
            [slot] if isinstance(slot, str) else slot for slot in slotList
        ]
        size += sum(
            get_size(getattr(obj, a, None), seen)
            for slot in slotList
            for a in slot
        )
    elif hasattr(obj, "__iter__") and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        size += sum(get_size(i, seen) for i in obj)
    return size


def check_memory():
    for name, size in sorted(
        ((name, size(value)) for name, value in locals().items()),
        key=lambda x: -x[1],
    )[:50]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)
