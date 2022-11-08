import numpy as np
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
from matplotlib import pyplot as plt

# define the figure size
small_figsize = (4, 3)  # default one, the previsous version was (8,6)
slide_figsize = (12, 6)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots notably
figsize = small_figsize


def plot_jaccard_aggregated(dic_jaccard, agg_periods, path):
    fig = plt.figure(figsize=figsize)
    length = len(dic_jaccard[str(agg_periods[0]) + " time steps"])
    for agg_period in agg_periods:
        plt.plot(
            np.arange(length),
            dic_jaccard[str(agg_period) + " time steps"],
        )

    plt.xlabel("Steps")
    plt.ylabel("Jaccard index")
    plt.title("Jaccard index aggregated")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in agg_periods],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "jaccard_index_agg.pdf", bbox_inches="tight")
    plt.close()


def plot_network_density(dic_density, agg_periods, path):
    fig = plt.figure(figsize=figsize)
    length = len(dic_density[str(agg_periods[0]) + " time steps"])
    for agg_period in agg_periods:
        plt.plot(
            np.arange(length),
            dic_density[str(agg_period) + " time steps"],
        )

    plt.xlabel("Steps")
    plt.ylabel("Network density")
    plt.title("Network density")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in agg_periods],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "network_density.pdf", bbox_inches="tight")
    plt.close()
