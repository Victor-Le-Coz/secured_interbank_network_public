import numpy as np
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
from matplotlib import pyplot as plt
import functions as fct
import pandas as pd
import graphics as gx

# define the figure size
small_figsize = (4, 3)  # default one, the previsous version was (8,6)
slide_figsize = (12, 6)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots notably
figsize = small_figsize


def plot_jaccard_aggregated(df_jaccard, path):
    fct.init_path(path)
    df_jaccard.to_csv(f"{path}df_jaccard.csv")
    df_jaccard.plot()
    plt.xlabel("Steps")
    plt.ylabel("Jaccard index")
    plt.title("Jaccard index aggregated")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in df_jaccard.keys()],
        loc="upper left",
    )
    plt.grid()
    plt.savefig(path + "jaccard_index_agg.pdf", bbox_inches="tight")
    plt.close()


def plot_network_density(df_density, path):
    fct.init_path(path)
    df_density.to_csv(f"{path}df_density.csv")
    df_density.plot()
    plt.xlabel("Steps")
    plt.ylabel("Network density")
    plt.title("Network density")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in df_density.keys()],
        loc="upper left",
    )
    plt.grid()
    plt.savefig(path + "network_density.pdf", bbox_inches="tight")
    plt.close()


def plot_step_degree_distribution(
    dic_in_degree,
    dic_out_degree,
    path,
    name,
    step,
):

    agg_periods = dic_in_degree.keys()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    width = 0.1
    space = 0.05
    pos = 0

    for agg_period in agg_periods:

        # in degree plot
        hist = np.unique(
            dic_in_degree[agg_period].loc[step],
            return_counts=True,
        )
        ax1.bar(hist[0] + pos, hist[1], width=width)

        # out degree plot
        hist = np.unique(
            dic_out_degree[agg_period].loc[step],
            return_counts=True,
        )
        ax2.bar(hist[0] + pos, hist[1], width=width)
        pos = pos + width + space

    ax1.set_xlabel("degree")
    ax1.set_ylabel("frequency")
    ax1.set_title("distribution of in-degree")
    ax1.legend(
        [str(agg_period) + " time steps" for agg_period in agg_periods],
        loc="upper left",
    )

    ax2.set_xlabel("degree")
    ax2.set_title("distribution of out-degree")
    ax2.legend(
        [str(agg_period) + " time steps" for agg_period in agg_periods],
        loc="upper left",
    )

    # ax2.set_xticks(hist[0] + (pos - width) / 2)
    # ax2.set_xticklabels(hist[0])

    plt.savefig(f"{path}{name}_{step}.pdf", bbox_inches="tight")
    plt.close()


def plot_degree_distribution(
    dic_in_degree, dic_out_degree, path, name, save_every
):
    fct.delete_n_init_path(path)
    for step in range(len(dic_in_degree[list(dic_in_degree.keys())[0]])):
        if step % save_every == 0:
            plot_step_degree_distribution(
                dic_in_degree,
                dic_out_degree,
                path,
                name,
                step,
            )

    agg_periods = dic_in_degree.keys()
    for agg_period in agg_periods:
        dic_in_degree[agg_period].to_csv(
            f"{path}df_in_degree_distribution_{agg_period}.csv"
        )
        dic_out_degree[agg_period].to_csv(
            f"{path}df_out_degree_distribution_{agg_period}.csv"
        )


def get_n_plot_cp_test(
    dic_obs_matrix_reverse_repo, algo, save_every, path_results
):

    sr_pvalue = pd.Series()

    fct.delete_n_init_path(path_results)

    for (step, ts_trade) in enumerate(dic_obs_matrix_reverse_repo.keys()):
        if step % save_every == 0:

            # build nx object
            bank_network = nx.from_numpy_array(
                np.asarray(dic_obs_matrix_reverse_repo[ts_trade]),
                parallel_edges=False,
                create_using=nx.DiGraph,
            )

            # run cpnet test
            sig_c, sig_x, significant, p_value = fct.cpnet_test(
                bank_network, algo=algo
            )

            # store the p_value
            sr_pvalue.loc[ts_trade] = p_value

            # plot
            gx.plot_core_periphery(
                bank_network=bank_network,
                sig_c=sig_c,
                sig_x=sig_x,
                path=f"{path_results}",
                step=ts_trade,
                name_in_title="reverse repo",
            )

    sr_pvalue.to_csv(f"{path_results}sr_pvalue.csv")
