import numpy as np
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
from matplotlib import pyplot as plt

# define the figure size
small_figsize = (4, 3)  # default one, the previsous version was (8,6)
slide_figsize = (12, 6)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots notably
figsize = small_figsize


def plot_jaccard_aggregated(dic_jaccard, path):
    fig = plt.figure(figsize=figsize)
    length = len(list(dic_jaccard.values())[0])
    for agg_period in dic_jaccard.keys():
        plt.plot(
            np.arange(length),
            dic_jaccard[agg_period],
        )

    plt.xlabel("Steps")
    plt.ylabel("Jaccard index")
    plt.title("Jaccard index aggregated")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in dic_jaccard.keys()],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "jaccard_index_agg.pdf", bbox_inches="tight")
    plt.close()


def plot_network_density(dic_density, path):
    fig = plt.figure(figsize=figsize)
    length = len(list(dic_density.values())[0])
    for agg_period in dic_density.keys():
        plt.plot(
            np.arange(length),
            dic_density[agg_period],
        )

    plt.xlabel("Steps")
    plt.ylabel("Network density")
    plt.title("Network density")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in dic_density.keys()],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "network_density.pdf", bbox_inches="tight")
    plt.close()


def plot_degree_distribution(
    dic_in_degree_distribution,
    dic_out_degree_distribution,
    path,
    name,
):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    width = 0.5
    pos = 0

    # in degree plot
    for agg_period in dic_in_degree_distribution.keys():
        hist = np.unique(
            dic_in_degree_distribution[agg_period][-1],
            return_counts=True,
        )
        ax1.bar(hist[0] + pos, hist[1], width=width)
        pos += width

    ax1.set_xlabel("degree")
    ax1.set_ylabel("frequency")
    ax1.set_title("distribution of in-degree")
    ax1.legend(
        [
            str(agg_period) + " time steps"
            for agg_period in dic_in_degree_distribution.keys()
        ],
        loc="upper left",
    )
    ax1.set_xticks(hist[0] + (pos - width) / 2)
    ax1.set_xticklabels(hist[0])

    # out degree plot
    pos = 0
    for agg_period in dic_out_degree_distribution.keys():
        hist = np.unique(
            dic_out_degree_distribution[agg_period][-1],
            return_counts=True,
        )
        ax2.bar(hist[0] + pos, hist[1], width=width)
        pos += width

    ax2.set_xlabel("degree")
    ax2.set_ylabel("frequency")
    ax2.set_title("distribution of out-degree")
    ax2.legend(
        [
            str(agg_period) + " time steps"
            for agg_period in dic_out_degree_distribution.keys()
        ],
        loc="upper left",
    )

    ax2.set_xticks(hist[0] + (pos - width) / 2)
    ax2.set_xticklabels(hist[0])

    fig.tight_layout()
    plt.savefig(path + name + ".pdf", bbox_inches="tight")
    plt.close()
