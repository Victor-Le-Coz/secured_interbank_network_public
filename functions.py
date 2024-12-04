import os
import numpy as np
import os
import shutil
import pandas as pd
from scipy import stats
import sys
import parameters as par
from tqdm import tqdm
from pathlib import Path


def build_args(dic_default_value, list_dic_range):
    list_dic_args = []
    for dic_range in list_dic_range:

        # build a dic of the dic_args with all the modif of the requested parameters
        dic_dic_arg_local = {}
        for arg, range in dic_range.items():

            for i, value in enumerate(range):

                # set the given arg to value
                if i in dic_dic_arg_local.keys():
                    dic_dic_arg_local[i][arg] = value
                    
                else:
                    # create a dic_args from the default values
                    dic_arg = dic_default_value.copy()

                    # modify one value
                    dic_arg[arg] = value

                    # store the dic_args created in a dic of dic args
                    dic_dic_arg_local.update({i:dic_arg})

        
        # manage the path
        arg, range = list(dic_range.items())[0]
        for i, value in enumerate(range):
            dic_dic_arg_local[i]["path_results"] = f"{dic_default_value['path_results']}{arg}/{i}#{value}/"

        # store each dic_args in a global list
        for value in dic_dic_arg_local.values():
            list_dic_args.append(value)
                    

    return list_dic_args

def get_dic_range(path):

    # get the list of input parameters
    repositories = os.listdir(path)

    # exclude from this list df_network_trajectory.csv if it exists
    input_parameters = [
        rep for rep in repositories if rep != "df_network_sensitivity.csv" and rep != "sr_errors.csv"
    ]

    dic_range = {}
    for input_parameter in input_parameters:
        # ar_range = np.sort(np.array(os.listdir(f"{path}{input_parameter}")))
        multi_range = os.listdir(f"{path}{input_parameter}")
        # transform the range in a dic of value as each value can appear several times
        dic_value = {}
        for value in multi_range:
            temp_list = value.split("#")
            value = temp_list[1]
            number = temp_list[0]
            if value in dic_value:
                dic_value[value].append(number)
            else:
                dic_value[value] = [number]
        dic_range.update({input_parameter: dic_value})
    return dic_range

def get_df_network_sensitivity(path):

    # get the input parameters and their ranges
    dic_range = get_dic_range(path)

    # initiaze the index of df_network_sensitivity
    index = pd.MultiIndex.from_tuples(
        [
            (input_parameter, float(value), int(number))
            for input_parameter in dic_range.keys() for value in dic_range[input_parameter].keys() for number in dic_range[input_parameter][value]
        ]
    )
    sr_errors = pd.Series(index=index)

    # fill-in df_network_sensitivity
    first_round = True
    for input_parameter in dic_range.keys():
        for value in tqdm(dic_range[input_parameter].keys()):
            for number in dic_range[input_parameter][value]:
                
                # collect the results
                path_df = (
                    f"{path}{input_parameter}/{number}#{value}/df_network_trajectory.csv"
                )
                if Path(f"{path}{input_parameter}/{number}#{value}/completed.txt").is_file():
                    if os.path.exists(path_df):

                        # load df_network_trajectory
                        try:
                            df_network_trajectory = pd.read_csv(path_df, index_col=0)
                        except:
                            continue

                        # initialize at the first round
                        if first_round:
                            df_network_sensitivity = pd.DataFrame(
                                index=index, columns=df_network_trajectory.columns
                            )
                            first_round = False

                        # fill with df_network_trajectory
                        df_network_sensitivity.loc[
                            (input_parameter, float(value), int(number))
                        ] = df_network_trajectory.iloc[-par.len_statio :].mean()
                    
                    
                # collect the errors
                path_error = f"{path}{input_parameter}/{number}#{value}/error.txt"
                if os.path.exists(path_error):
                    sr_errors.loc[
                        (input_parameter, float(value), int(number))
                    ] = open(path_error, "r").read()

    # save the results
    df_network_sensitivity.to_csv(f"{path}/df_network_sensitivity.csv")
    sr_errors.to_csv(f"{path}/sr_errors.csv")

    return df_network_sensitivity, sr_errors


def get_plot_steps_from_days(days, plot_days):
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


def get_plot_days_from_period(days, plot_period):
    nb_days = len(days)
    plot_days = [
        days[step]
        for step in range(len(days))
        if step % plot_period == 0 or step == nb_days - 1
    ]
    return plot_days


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
        try:
            bl_len = len(obj) > 0
        except:
            bl_len = False

        if bl_len:
            size += sum(get_size(i, seen) for i in obj)
        else:
            size = 0

    return size


def check_memory():
    # sizes = []
    # items = list(locals().items())[:10]
    # for name, value in items:
    #     try:
    #         size = sizeof_fmt(get_size(value))
    #         sizes.append(name, size)
    #     except:
    #         pass

    # for size in sizes:
    #     print("{:>30}: {:>8}".format(size[0], size[1]))

    for (name, size) in sorted(
        ((name, get_size(value)) for name, value in locals().items()),
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


def list_exclusion(list1, list2):
    return [x for x in list1 if x not in set(list2)]


def filtered_mean(group):
    mean = group.mean()
    std = group.std()
    filtered_group = group[(group >= mean - std) & (group <= mean + std)]
    return filtered_group.mean()