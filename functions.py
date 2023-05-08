import os

# os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import os
import shutil
import pandas as pd
from scipy import stats
import sys


def build_args(dic_default_value, dic_ranges):

    list_dic_args = []
    for arg, range in dic_ranges.items():
        for value in range:

            # create a dic_args from the default values
            dic_args = dic_default_value.copy()

            # set the given arg to value and the path to arg/value/
            dic_args[arg] = value
            dic_args[
                "result_location"
            ] = f"{dic_args['result_location']}{arg}/{value}/"

            # specific case of beta
            if arg == "beta_reg":
                dic_args["beta_init"] = value
                dic_args["beta_star"] = value

            # call the function with the current parameter value
            list_dic_args.append(dic_args)

    return list_dic_args


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


def list_intersection(list1, list2):
    return [x for x in list1 if x in set(list2)]


def get_plot_steps_from_days(days, plot_days):
    nb_days = len(days)
    plot_steps = [step for step in range(len(days)) if days[step] in plot_days]
    return plot_steps


def get_plot_steps_from_period(days, plot_period):
    nb_days = len(days)
    plot_steps = [
        step
        for step in range(len(days))
        if step % plot_period == 0 or step == nb_days - 1
    ]
    return plot_steps
