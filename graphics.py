from cProfile import label
import os
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import function as fct


def bar_plot_deposits(deposits, path, step):
    plt.figure(figsize=(20, 10))
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
    plt.ylabel("Relative Deposits in %", fontweight="bold", fontsize=15)
    plt.xticks([r for r in range(len(deposits))], banks_sorted)
    plt.tick_params(axis="x", labelrotation=90, labelsize="small")
    plt.legend()
    plt.title("Deposits of Banks at step {}".format(int(step)))
    plt.savefig(os.path.join(path, "step_{}_deposits.png".format(step)))
    plt.close()


def bar_plot_balance_sheet(
    total_assets,
    network_assets,
    network_liabilities,
    network_off_balance,
    path,
    step,
):
    plt.figure(figsize=(30, 15))
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_figheight(15)
    fig.set_figwidth(30)

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
        ]
    )
    ax1.tick_params(axis="x", labelrotation=90, labelsize="small")
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
    ax2.legend(["Own Funds", "Deposits", "Repos", "MROs"])
    ax2.tick_params(axis="x", labelrotation=90, labelsize="small")
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
    ax3.legend(["Securities Collateral", "Securities Reused"])
    ax3.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax3.set_title("Off Balance Sheets of Banks at step {}".format(int(step)))

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(path, "step_{}_balance_sheets.png".format(step)))
    plt.close()


def plot_assets_loans_mros(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Securities Usable"])
    plt.plot(np.arange(length), time_series_metrics["Loans"])
    plt.plot(np.arange(length), time_series_metrics["MROs"])
    plt.plot(np.arange(length), time_series_metrics["Assets"])
    plt.legend(["Loans", "MROs", "Assets"])
    plt.xlabel("Steps")
    plt.ylabel("Total Network Amount")
    plt.title("Total Amount of Assets, Loans, and MROs in Network")
    plt.savefig(os.path.join(path, "Assets_loans_mros.png"))
    plt.close()


def plot_network_density(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Network Density"])
    plt.plot(np.arange(length), time_series_metrics["Network Density"])
    plt.xlabel("Steps")
    plt.ylabel("Density")
    plt.title("Network Density across time")
    plt.savefig(os.path.join(path, "network_density.png"))
    plt.close()


def plot_gini(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Gini"])
    plt.plot(np.arange(length), time_series_metrics["Gini"])
    plt.xlabel("Steps")
    plt.ylabel("Gini")
    plt.title("Gini coefficient across time of the total assets per bank")
    plt.savefig(os.path.join(path, "gini.png"))
    plt.close()


def plot_reverse_repo_size_stats(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Reverse repo size min"])
    plt.plot(np.arange(length), time_series_metrics["Reverse repo size min"])
    # plt.plot(np.arange(length), time_series_metrics["Reverse repo size max"])
    # plt.plot(np.arange(length), time_series_metrics["Reverse repo size mean"])
    plt.xlabel("Steps")
    plt.ylabel("Reverse repo size stats")
    # plt.gca().set_yscale("log")
    plt.legend(
        [
            "min",
            # "max",
            # "mean",
        ]
    )
    plt.title("Reverse repo size statistics across time")
    plt.savefig(os.path.join(path, "reverse_repo_stats.png"))
    plt.close()


def plot_collateral_reuse(reuse, path):
    plt.figure()
    length = len(reuse)
    plt.plot(np.arange(length), reuse)
    plt.xlabel("Steps")
    plt.ylabel("Reused/Total Collateral")
    plt.title("Reuse of collateral over time")
    plt.savefig(os.path.join(path, "collateral_reuse.png"))
    plt.close()


def plot_repos(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Repos"])
    plt.plot(np.arange(length), time_series_metrics["Repos"])
    plt.plot(np.arange(length), time_series_metrics["Reverse Repos"])
    plt.legend(["Repos", "Reverse Repos"])
    plt.xlabel("Steps")
    plt.ylabel("Total Network Amount")
    plt.title("Total Amount of Repos/Reverse Repos")
    plt.savefig(os.path.join(path, "Repos.png"))
    plt.close()


def plot_jaccard(time_series_metrics, period, path):
    plt.figure()
    length = len(time_series_metrics["Jaccard Index"])
    plt.plot(np.arange(length), time_series_metrics["Jaccard Index"])
    plt.xlabel("Steps")
    plt.ylabel("Jaccard Index")
    plt.title(
        "Temporal Developpement of Jaccard Index for {} period, \n final value is {}".format(
            period, np.mean(time_series_metrics["Jaccard Index"][-50:])
        )
    )
    plt.grid()
    plt.yticks(np.arange(0, 1, 0.05))
    plt.savefig(os.path.join(path, "jaccard_index.png"))
    plt.close()


def plot_excess_liquidity_and_deposits(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Excess Liquidity"])
    plt.plot(np.arange(length), time_series_metrics["Excess Liquidity"])
    plt.plot(np.arange(length), time_series_metrics["Deposits"])
    plt.legend(["Excess Liquidity", "Deposits"])
    plt.xlabel("Steps")
    plt.ylabel("Total Network Amount")
    plt.title("Total Excess Liquidity and total Deposits")
    plt.savefig(os.path.join(path, "excess_liquidity_and_deposits.png"))
    plt.close()


def plot_collateral(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Securities Usable"])
    plt.plot(np.arange(length), time_series_metrics["Securities Usable"])
    plt.plot(
        np.arange(length),
        time_series_metrics["Securities Encumbered"],
    )
    plt.plot(
        np.arange(length),
        time_series_metrics["Securities Collateral"],
    )
    plt.plot(np.arange(length), time_series_metrics["Securities Reused"])
    plt.legend(
        [
            "Securities Usable",
            "Securities Encumbered",
            "Securities Collateral",
            "Securities Reused",
        ]
    )
    plt.xlabel("Steps")
    plt.ylabel("Total Network Amount")
    plt.title("Total Amount of Collateral in Network")
    plt.savefig(os.path.join(path, "collateral.png"))
    plt.close()


def plot_degre_network(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["In-degree"])
    plt.plot(np.arange(length), time_series_metrics["In-degree"])
    plt.xlabel("Steps")
    plt.ylabel("Average in-degree")
    plt.title("Average in-degree in the repo network")
    plt.savefig(os.path.join(path, "average_in-degree.png"))
    plt.close()


def plot_average_nb_transactions(time_series_metrics, path):
    plt.figure()
    length = len(
        time_series_metrics[
            "Average number of repo transaction ended within a step"
        ]
    )
    plt.plot(
        np.arange(length),
        time_series_metrics[
            "Average number of repo transaction ended within a step"
        ],
    )
    plt.xlabel("Steps")
    plt.ylabel("Number of transactions")
    plt.title(
        "Average number of repo transaction ended within a step in the network"
    )
    plt.savefig(os.path.join(path, "Average_nb_repo_transactions_ended.png"))
    plt.close()


def plot_average_size_transactions(time_series_metrics, path):
    plt.figure()
    length = len(
        time_series_metrics[
            "Average size of repo transaction ended within a step"
        ]
    )
    plt.plot(
        np.arange(length),
        time_series_metrics[
            "Average size of repo transaction ended within a step"
        ],
    )
    plt.xlabel("Steps")
    plt.ylabel("Size of transactions")
    plt.title(
        "Average size of repo transaction ended within a step in the network"
    )
    plt.savefig(os.path.join(path, "Average_size_repo_transactions_ended.png"))
    plt.close()


def plot_average_maturity_repo(time_series_metrics, path):
    plt.figure()
    length = len(time_series_metrics["Average maturity of repos"])
    plt.plot(
        np.arange(length),
        time_series_metrics["Average maturity of repos"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Weighted average maturity of repos")
    plt.title("Weighted average maturity of repos")
    plt.savefig(os.path.join(path, "Average_maturity_repo.png"))
    plt.close()


def plot_network(adj, path, step, name):
    # build a network from an adjacency matrix
    bank_network = nx.from_numpy_matrix(
        adj, parallel_edges=False, create_using=nx.DiGraph
    )
    # define the weight list from the weight information
    weights = [
        bank_network[node1][node2]["weight"]
        for node1, node2 in bank_network.edges()
    ]

    # define the position of the nodes
    pos = nx.spring_layout(bank_network)

    # draw the network
    plt.figure(1, figsize=(15, 15))
    nx.draw_networkx(bank_network, pos, width=weights, with_labels=True)

    # show the plot
    plt.title("{} network at the step {}".format(name, int(step)))
    plt.savefig(os.path.join(path, "step_{}_network.png".format(step)))
    plt.close()


def plot_core_periphery(bank_network, sig_c, sig_x, path, step, name):
    # Visualization
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax, pos = cpnet.draw(bank_network, sig_c, sig_x, ax)

    # show the plot
    plt.title(
        "{} core-periphery structure at the step {}".format(name, int(step))
    )
    plt.savefig(
        os.path.join(
            path,
            "step_{" "}_core-periphery_structure.png".format(step),
        )
    )
    plt.close()


def plot_asset_per_degree(total_assets, degree, path):
    plt.plot(degree, total_assets, ".")
    plt.xlabel("Degree")
    plt.ylabel("Total assets")
    plt.title(
        "Total assets per bank as a fonction of the degree in the network"
    )
    plt.savefig(os.path.join(path, "Asset_per_degree.png"))
    plt.close()


def plot_single_trajectory(single_trajectory, path):

    # Plot the accounting items
    keys = [
        "Cash",
        "Securities Usable",
        "Securities Encumbered",
        # "Loans",
        "Reverse Repos",
        "Own Funds",
        # "Deposits",
        "Repos",
        # "MROs",
        "Securities Collateral",
        "Securities Reused",
    ]

    plt.figure(figsize=(30, 15))
    length = len(single_trajectory["Cash"])
    for key in keys:

        # Log scale
        # plt.plot(
        #     np.arange(length), np.log10(np.abs(single_trajectory[key])+1),
        #     label=str(key)
        # )

        # usual scale
        plt.plot(np.arange(length), single_trajectory[key], label=str(key))

    plt.xlabel("Steps")
    # plt.ylabel("Log10 of monetary units")
    plt.ylabel("Monetary units")
    plt.legend(loc="upper left")
    plt.grid()
    plt.title("Single bank trajectory of accounting items")
    plt.savefig(os.path.join(path, "Single_trajectory_accounting_items.png"))
    plt.close()

    # Plot the other indicators
    keys = [
        "In-degree",
        "Out-degree",
        "Number of repo transaction ended within a step",
        "Size of repo transaction ended within a step",
        "Maturity of repos",
    ]
    plt.figure(figsize=(30, 15))
    length = len(single_trajectory["In-degree"])
    for key in keys:
        plt.plot(np.arange(length), single_trajectory[key], label=str(key))

    plt.xlabel("Steps")
    plt.ylabel("Indicators")
    plt.legend(loc="upper left")
    plt.grid()
    plt.title("Single bank trajectory of indicators")
    plt.savefig(os.path.join(path, "Single_trajectory_indicators.png"))
    plt.close()


def plot_output_by_args(args, axe, output, path):
    output = fct.reformat_output(output)
    for key in output.keys():
        plt.plot(args, output[key], "-o")
        plt.xlabel(axe)
        if axe == "min_repo_size":
            plt.gca().set_xscale("log")
        plt.ylabel(key)
        plt.title(key + " as a function of " + axe)
        plt.savefig(os.path.join(path, key + "_" + axe + ".png"))
        plt.close()
