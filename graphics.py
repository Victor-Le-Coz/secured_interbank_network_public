import os

os.environ["OMP_NUM_THREADS"] = "1"
from cProfile import label
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import function as fct

# define the figure size for all banks
small_figsize = (4, 3)  # default one, the previsous version was (8,6)
slide_figsize = (12, 6)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots notably
figsize = small_figsize


def bar_plot_deposits(deposits, path, step):
    fig = plt.figure(figsize=halfslide_figsize)
    # banks_sorted = np.argsort(deposits)
    banks_sorted = ["Bank {}".format(str(b)) for b in range(len(deposits))]
    deposits_sorted = deposits.copy()
    # deposits_sorted = np.sort(deposits)
    # deposits_sorted = deposits_sorted / deposits_sorted.sum()

    barWidth = 0.75
    br = np.arange(len(deposits))
    plt.bar(
        br,
        height=deposits_sorted,
        color="b",
        width=barWidth,
        edgecolor="grey",
        label="Deposits",
    )
    plt.ylabel("Deposits", fontweight="bold", fontsize=15)
    plt.xticks([r for r in range(len(deposits))], banks_sorted)
    plt.tick_params(axis="x", labelrotation=90, labelsize="small")
    plt.legend(loc="upper left")
    plt.title("Deposits of Banks at step {}".format(int(step)))
    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "step_{}_deposits.pdf".format(step)),
        bbox_inches="tight",
    )
    plt.close()


def bar_plot_balance_sheet(
    total_assets,
    network_assets,
    network_liabilities,
    network_off_balance,
    path,
    step,
):
    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=halfslide_figsize)

    ix_sorted = np.argsort(total_assets)
    banks_sorted = ["Bank {}".format(str(b)) for b in ix_sorted]

    a1 = network_assets["Cash"][ix_sorted]
    a2 = a1 + network_assets["Securities Usable"][ix_sorted]
    a3 = a2 + network_assets["Securities Encumbered"][ix_sorted]
    a4 = a3 + network_assets["Loans"][ix_sorted]
    a5 = a4 + network_assets["Reverse Repos"][ix_sorted]

    b1 = network_liabilities["Own Funds"][ix_sorted]
    b2 = b1 + network_liabilities["Deposits"][ix_sorted]
    b3 = b2 + network_liabilities["Repos"][ix_sorted]
    b4 = b3 + network_liabilities["MROs"][ix_sorted]

    barWidth = 0.75

    ax1.bar(
        banks_sorted,
        height=a1,
        color="cyan",
        width=barWidth,
        label="Cash",
    )
    ax1.bar(
        banks_sorted,
        height=network_assets["Securities Usable"][ix_sorted],
        bottom=a1,
        color="green",
        width=barWidth,
        label="Securities Usable",
    )
    ax1.bar(
        banks_sorted,
        height=network_assets["Securities Encumbered"][ix_sorted],
        bottom=a2,
        color="red",
        width=barWidth,
        label="Securities Encumbered",
    )
    ax1.bar(
        banks_sorted,
        height=network_assets["Loans"][ix_sorted],
        bottom=a3,
        color="blue",
        width=barWidth,
        label="Loans",
    )
    ax1.bar(
        banks_sorted,
        height=network_assets["Reverse Repos"][ix_sorted],
        bottom=a4,
        color="yellow",
        width=barWidth,
        label="Reverse Repos",
    )
    ax1.legend(
        [
            "Cash",
            "Securities Usable",
            "Securities Encumbered",
            "Loans",
            "Reverse Repos",
        ],
        loc="upper left",
    )
    ax1.tick_params(
        axis="x",
        labelrotation=90,
        labelsize="small",
        bottom=False,
        top=False,
        labelbottom=False,
        which="both",
    )
    ax1.set_title("Assets of Banks at step {}".format(int(step)))

    ax2.bar(
        banks_sorted,
        height=b1,
        color="cyan",
        width=barWidth,
        label="Own Funds",
    )
    ax2.bar(
        banks_sorted,
        height=network_liabilities["Deposits"][ix_sorted],
        bottom=b1,
        color="green",
        width=barWidth,
        label="Deposits",
    )
    ax2.bar(
        banks_sorted,
        height=network_liabilities["Repos"][ix_sorted],
        bottom=b2,
        color="red",
        width=barWidth,
        label="Repos",
    )
    ax2.bar(
        banks_sorted,
        height=network_liabilities["MROs"][ix_sorted],
        bottom=b3,
        color="blue",
        width=barWidth,
        label="MROs",
    )
    ax2.legend(["Own Funds", "Deposits", "Repos", "MROs"], loc="upper left")
    ax2.tick_params(
        axis="x",
        labelrotation=90,
        labelsize="small",
        bottom=False,
        top=False,
        labelbottom=False,
        which="both",
    )
    ax2.set_title("Liabilities of Banks at step {}".format(int(step)))

    ax3.bar(
        banks_sorted,
        height=network_off_balance["Securities Collateral"][ix_sorted],
        color="green",
        width=barWidth,
        label="Own Funds",
    )
    ax3.bar(
        banks_sorted,
        height=network_off_balance["Securities Reused"][ix_sorted],
        bottom=network_off_balance["Securities Collateral"][ix_sorted],
        color="red",
        width=barWidth,
        label="Own Funds",
    )
    ax3.legend(["Securities Collateral", "Securities Reused"], loc="upper left")
    ax3.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax3.set_title("Off Balance Sheets of Banks at step {}".format(int(step)))

    # plt.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "step_{}_balance_sheets.pdf".format(step)),
        bbox_inches="tight",
    )
    plt.close()


def plot_assets_loans_mros(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Securities Usable tot. volume"])
    plt.plot(np.arange(length), time_series_metrics["Loans tot. volume"])
    plt.plot(np.arange(length), time_series_metrics["MROs tot. volume"])
    plt.plot(np.arange(length), time_series_metrics["Assets tot. volume"])
    # plt.plot(np.arange(length), time_series_metrics["Deposits"])
    plt.legend(["Loans", "MROs", "Assets"])
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total assets, loans, and MROs")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "Assets_loans_mros.pdf"), bbox_inches="tight")
    plt.close()


def plot_network_density(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Network density"])
    plt.plot(np.arange(length), time_series_metrics["Network density"])
    plt.xlabel("Steps")
    plt.ylabel("Density")
    plt.title("Network density")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "network_density.pdf"), bbox_inches="tight")
    plt.close()


def plot_gini(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Gini"])
    plt.plot(np.arange(length), time_series_metrics["Gini"])
    plt.xlabel("Steps")
    plt.ylabel("Gini coefficient")
    plt.title("Gini of banks' assets")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "gini.pdf"), bbox_inches="tight")
    plt.close()


def plot_reverse_repo_size_stats(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Repos min volume"])
    plt.plot(np.arange(length), time_series_metrics["Repos min volume"])
    plt.plot(np.arange(length), time_series_metrics["Repos max volume"])
    plt.plot(np.arange(length), time_series_metrics["Repos av. volume"])
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.legend(
        [
            "min",
            "max",
            "mean",
        ],
        loc="upper left",
    )

    plt.title("Repo amount stats")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "reverse_repo_stats.pdf"), bbox_inches="tight")
    plt.close()


def plot_collateral_reuse(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Collateral reuse"])
    plt.plot(np.arange(length), time_series_metrics["Collateral reuse"])
    plt.xlabel("Steps")
    plt.ylabel("Ratio")
    plt.title("Collateral reuse")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "collateral_reuse.pdf"), bbox_inches="tight")
    plt.close()


def plot_repos(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Repos tot. volume"])
    plt.plot(np.arange(length), time_series_metrics["Repos tot. volume"])
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total repo amount")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "Repos_market_size.pdf"), bbox_inches="tight")
    plt.close()


def plot_jaccard(time_series_metrics, jaccard_periods, path):
    fig = plt.figure(figsize=figsize)
    length = len(
        time_series_metrics["Jaccard index " + str(jaccard_periods[0]) + " time steps"]
    )
    for jaccard_period in jaccard_periods:
        plt.plot(
            np.arange(length),
            time_series_metrics["Jaccard index " + str(jaccard_period) + " time steps"],
        )

    plt.xlabel("Steps")
    plt.ylabel("Jaccard index")
    plt.title("Jaccard index")
    plt.legend(
        [str(jaccard_period) + " time steps" for jaccard_period in jaccard_periods],
        loc="upper left",
    )
    plt.grid()
    # plt.yticks(np.arange(0, 1, 0.05))
    fig.tight_layout()
    plt.savefig(os.path.join(path, "jaccard_index.pdf"), bbox_inches="tight")
    plt.close()


def plot_excess_liquidity_and_deposits(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Excess Liquidity"])
    plt.plot(np.arange(length), time_series_metrics["Excess Liquidity"])
    plt.plot(np.arange(length), time_series_metrics["Deposits tot. volume"])
    plt.legend(["Excess Liquidity", "Deposits"], loc="upper left")
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total excess liquidity & deposits")
    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "excess_liquidity_and_deposits.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_collateral(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Securities Usable tot. volume"])
    plt.plot(np.arange(length), time_series_metrics["Securities Usable tot. volume"])
    plt.plot(
        np.arange(length),
        time_series_metrics["Securities Encumbered tot. volume"],
    )
    plt.plot(
        np.arange(length),
        time_series_metrics["Securities Collateral tot. volume"],
    )
    plt.plot(np.arange(length), time_series_metrics["Securities Reused tot. volume"])
    plt.legend(
        [
            "Securities Usable",
            "Securities Encumbered",
            "Securities Collateral",
            "Securities Reused",
        ],
        loc="upper left",
    )
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total collateral")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "collateral.pdf"), bbox_inches="tight")
    plt.close()


def plot_degre_network(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Av. in-degree"])
    plt.plot(np.arange(length), time_series_metrics["Av. in-degree"])
    plt.xlabel("Steps")
    plt.ylabel("Av. in-degree")
    plt.title("Av. in-degree")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "average_in-degree.pdf"), bbox_inches="tight")
    plt.close()


def plot_average_nb_transactions(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Av. nb. of repo transactions ended"])
    plt.plot(
        np.arange(length),
        time_series_metrics["Av. nb. of repo transactions ended"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Number of transactions")
    plt.title("Av. nb. of repo transactions ended")
    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "Average_nb_repo_transactions_ended.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_average_size_transactions(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Av. volume of repo transactions ended"])
    plt.plot(
        np.arange(length),
        time_series_metrics["Av. volume of repo transactions ended"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Av. volume of repo transactions ended")
    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "Average_size_repo_transactions_ended.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_average_maturity_repo(time_series_metrics, path):
    fig = plt.figure(figsize=figsize)
    length = len(time_series_metrics["Repos av. maturity"])
    plt.plot(
        np.arange(length),
        time_series_metrics["Repos av. maturity"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Maturity")
    plt.title("Repos av. maturity")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "Average_maturity_repo.pdf"), bbox_inches="tight")
    plt.close()


def plot_network(adj, network_total_assets, path, step, name):
    # build a network from an adjacency matrix
    bank_network = nx.from_numpy_matrix(
        adj, parallel_edges=False, create_using=nx.DiGraph
    )
    # define the weight list from the weight information
    weights = [
        bank_network[node1][node2]["weight"] for node1, node2 in bank_network.edges()
    ]
    # log scale the big values in the repo network
    log_weights = [0 if i <= 1 else np.log(i) + 1 for i in weights]

    # define the size of the nodes a a function of the total deposits
    node_sizes = network_total_assets

    # define the position of the nodes
    # pos = nx.spring_layout(bank_network)
    pos = nx.circular_layout(bank_network)

    # draw the network
    fig = plt.figure(figsize=small_figsize)
    nx.draw_networkx(
        bank_network,
        pos,
        width=log_weights,
        with_labels=True,
        node_size=node_sizes,
    )

    # show the plot
    plt.title("{} network at the step {}".format(name, int(step)))
    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "step_{}_network.pdf".format(step)),
        bbox_inches="tight",
    )
    plt.close()


def plot_core_periphery(bank_network, sig_c, sig_x, path, step, name):
    # Visualization
    fig = plt.figure(figsize=halfslide_figsize)
    ax = plt.gca()
    ax, pos = cpnet.draw(bank_network, sig_c, sig_x, ax)

    # show the plot
    plt.title("{} core-periphery structure at the step {}".format(name, int(step)))
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            path,
            "step_{" "}_core-periphery_structure.pdf".format(step),
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_asset_per_degree(total_assets, degree, path):
    fig = plt.figure(figsize=figsize)
    plt.scatter(degree, total_assets)
    for i in range(len(degree)):
        plt.text(x=degree[i], y=total_assets[i], s=str(i))
    plt.xlabel("Degree")
    plt.ylabel("Assets")
    plt.title("Assets size verus degree")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "Asset_per_degree.pdf"), bbox_inches="tight")
    plt.close()


def plot_single_trajectory(single_trajectory, path):

    plt.figure(figsize=figsize)
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=slide_figsize)

    # Plot the accounting items (big)
    keys_s = [
        "Loans",
        "Deposits",
        "MROs",
    ]

    length = len(single_trajectory["Loans"])
    for key in keys_s:
        ax1.plot(np.arange(length), single_trajectory[key], label=str(key))

    ax1.tick_params(
        axis="x",
        bottom=False,
        top=False,
        labelbottom=False,
        which="both",
    )
    ax1.set_ylabel("Monetary units")
    ax1.legend(loc="upper left")
    ax1.grid()
    ax1.set_title("Single bank trajectory of (large) accounting items")

    # Plot the accounting items (small)
    keys_s = [
        "Cash",
        "Securities Usable",
        "Securities Encumbered",
        # "Reverse Repos",
        # "Own Funds",
        # "Repos",
        "Securities Collateral",
        "Securities Reused",
    ]

    for key in keys_s:
        ax2.plot(np.arange(length), single_trajectory[key], label=str(key))

    ax2.tick_params(
        axis="x",
        bottom=False,
        top=False,
        labelbottom=False,
        which="both",
    )
    ax2.set_ylabel("Monetary units")
    ax2.legend(loc="upper left")
    ax2.grid()
    ax2.set_title("Single bank trajectory of (small) accounting items")

    # Plot the other indicators
    keys_s = [
        "Av. in-degree",
        "Av. out-degree",
        "Nb. of repo transactions ended",
        # "Av. volume of repo transactions ended",
        "Repos av. maturity",
    ]
    for key in keys_s:
        ax3.plot(np.arange(length), single_trajectory[key], label=str(key))

    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Indicators")
    ax3.legend(loc="upper left")
    ax3.grid()
    ax3.set_title("Single bank trajectory of indicators")

    fig.tight_layout()
    plt.savefig(
        os.path.join(path, "Single_trajectory_indicators.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_output_by_args(args, axe, output, path):
    output = fct.reformat_output(output)

    # linear chart
    for key in output.keys():  # plot all jaccard on same chart
        if key[0:13] == "Jaccard index":
            fig = plt.figure(figsize=figsize)
            plt.plot(args, output[key], "-o")
            if axe == "min_repo_size" or axe == "alpha_pareto" or axe == "shocks_vol":
                plt.gca().set_xscale("log")
                plt.xlabel(axe + " (log-scale)")
            else:
                plt.xlabel(axe)
            plt.ylabel(key)
            plt.title(key + " as a fct. of " + axe)
            fig.tight_layout()
    plt.savefig(
        os.path.join(path, "Jaccard index" + "_" + axe + ".pdf"),
        bbox_inches="tight",
    )
    plt.close()
    for key in output.keys():  # plot all separated charts
        fig = plt.figure(figsize=figsize)
        plt.plot(args, output[key], "-o")
        if axe == "min_repo_size" or axe == "alpha_pareto" or axe == "shocks_vol":
            plt.gca().set_xscale("log")
            plt.xlabel(axe + " (log-scale)")
        else:
            plt.xlabel(axe)
        plt.ylabel(key)
        plt.title(key + " as a fct. of " + axe)
        fig.tight_layout()
        plt.savefig(os.path.join(path, key + "_" + axe + ".pdf"), bbox_inches="tight")
        plt.close()

    # log chart - to answer Michael's comment when required
    for key in output.keys():
        fig = plt.figure(figsize=figsize)
        plt.plot(args, output[key], "-o")
        plt.xlabel(axe + " (log-scale)")
        plt.gca().set_xscale("log")
        plt.ylabel(key + " (log-scale)")
        plt.gca().set_yscale("log")
        plt.title(key + " as a fct. of " + axe)
        fig.tight_layout()
        plt.savefig(
            os.path.join(path + "log_scale/", "log_" + key + "_" + axe + ".pdf"),
            bbox_inches="tight",
        )
        plt.close()
