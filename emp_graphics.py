import numpy as np
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
from matplotlib import pyplot as plt
import functions as fct
import pandas as pd
import parameters as par


def plot_core_periphery(
    bank_network,
    sig_c,
    sig_x,
    path,
    step,
    name_in_title,
    figsize=(6, 3),
):

    fct.init_path(path)

    # Visualization
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax, pos = cpnet.draw(bank_network, sig_c, sig_x, ax)

    # show the plot
    plt.title(
        "{} core-periphery structure at the step {}".format(
            name_in_title, step
        )
    )
    fig.tight_layout()
    plt.savefig(
        f"{path}step_core-periphery_structure_{step}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def run_n_plot_cp_test(
    arr_matrix_reverse_repo,
    algo,
    plot_every,
    days,
    path_results,
    figsize=(6, 3),
):

    print(f"core-periphery tests using the {algo} approach")

    # initialise results and path
    sr_pvalue = pd.Series()
    fct.delete_n_init_path(path_results)

    for step, day in enumerate(days[1:], 1):

        # we run the analyis every x days + for the last day
        if step % plot_every == 0 or day == days[-1]:

            print(f"test at step {step}")

            # build nx object
            bank_network = nx.from_numpy_array(
                arr_matrix_reverse_repo[step],
                parallel_edges=False,
                create_using=nx.DiGraph,
            )

            # run cpnet test
            sig_c, sig_x, significant, p_values = fct.cpnet_test(
                bank_network, algo=algo
            )

            # store the p_value (only the first one)
            sr_pvalue.loc[day] = p_values[0]

            # plot
            if isinstance(day, pd.Timestamp):
                day_print = day.strftime("%Y-%m-%d")
            else:
                day_print = day

            plot_core_periphery(
                bank_network=bank_network,
                sig_c=sig_c,
                sig_x=sig_x,
                path=f"{path_results}",
                step=day_print,
                name_in_title="reverse repo",
                figsize=figsize,
            )

    sr_pvalue.to_csv(f"{path_results}sr_pvalue.csv")

    return sr_pvalue


def mlt_run_n_plot_cp_test(
    dic,
    algos,
    plot_every,
    days,
    path_results,
    figsize=(6, 3),
    opt_agg=False,
):

    print("run core-periphery tests")

    # case dijonction to build the list of agg periods
    if opt_agg:
        agg_periods = dic.keys()
    else:
        agg_periods = ["weighted"]

    for agg_period in agg_periods:

        # define the path
        path = f"{path_results}core-periphery/{agg_period}/"

        # case dijonction for the dictionary of adjency periods
        if opt_agg:
            dic_adj = dic[agg_period]
        else:
            dic_adj = dic

        df_pvalue = pd.DataFrame(columns=algos)
        for algo in algos:
            df_pvalue[algo] = run_n_plot_cp_test(
                dic_adj,
                algo=algo,
                plot_every=plot_every,
                days=days,
                path_results=f"{path}{algo}/",
                figsize=figsize,
            )
        df_pvalue.to_csv(f"{path}df_pvalue.csv")

        ax = df_pvalue.plot(figsize=figsize, style=".")
        lgd = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(
            f"{path}pvalues.pdf",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.close()
