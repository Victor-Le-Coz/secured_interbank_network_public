# import librairies
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from cProfile import label
import cpnet  # Librairy for the estimation of core-periphery structures
import networkx as nx
from matplotlib import pyplot as plt

# import modules
import functions as fct
import parameters as par

# define the figure size
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
        label="deposits",
    )
    plt.ylabel("deposits", fontweight="bold", fontsize=15)
    plt.xticks([r for r in range(len(deposits))], banks_sorted)
    plt.tick_params(axis="x", labelrotation=90, labelsize="small")
    plt.legend(loc="upper left")
    plt.title("Deposits of Banks at step {}".format(int(step)))
    fig.tight_layout()
    plt.savefig(
        path + "step_{}_deposits.pdf".format(step),
        bbox_inches="tight",
    )
    plt.close()


def bar_plot_balance_sheet(
    df_banks,
    path,
    step,
):
    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=halfslide_figsize)

    ix_sorted = np.argsort(np.asarray(df_banks["total assets"]))
    banks_sorted = ["Bank {}".format(str(b)) for b in ix_sorted]

    a1 = df_banks.loc[ix_sorted, "cash"]
    a2 = a1 + df_banks.loc[ix_sorted, "securities usable"]
    a3 = a2 + df_banks.loc[ix_sorted, "securities encumbered"]
    a4 = a3 + df_banks.loc[ix_sorted, "loans"]
    a5 = a4 + df_banks.loc[ix_sorted, "reverse repo exposures"]

    b1 = df_banks.loc[ix_sorted, "own funds"]
    b2 = b1 + df_banks.loc[ix_sorted, "deposits"]
    b3 = b2 + df_banks.loc[ix_sorted, "repo exposures"]
    b4 = b3 + df_banks.loc[ix_sorted, "central bank funding"]

    barWidth = 0.75

    ax1.bar(
        banks_sorted,
        height=a1,
        color="cyan",
        width=barWidth,
        label="cash",
    )
    ax1.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "securities usable"],
        bottom=a1,
        color="green",
        width=barWidth,
        label="securities usable",
    )
    ax1.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "securities encumbered"],
        bottom=a2,
        color="red",
        width=barWidth,
        label="securities encumbered",
    )
    ax1.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "loans"],
        bottom=a3,
        color="blue",
        width=barWidth,
        label="loans",
    )
    ax1.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "reverse repo exposures"],
        bottom=a4,
        color="yellow",
        width=barWidth,
        label="reverse repo exposures",
    )
    ax1.legend(
        [
            "cash",
            "securities usable",
            "securities encumbered",
            "loans",
            "reverse repo exposures",
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
        label="own funds",
    )
    ax2.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "deposits"],
        bottom=b1,
        color="green",
        width=barWidth,
        label="deposits",
    )
    ax2.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "repo exposures"],
        bottom=b2,
        color="red",
        width=barWidth,
        label="repo exposures",
    )
    ax2.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "central bank funding"],
        bottom=b3,
        color="blue",
        width=barWidth,
        label="central bank funding",
    )
    ax2.legend(
        ["own funds", "deposits", "repo exposures", "central bank funding"],
        loc="upper left",
    )
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
        height=df_banks.loc[ix_sorted, "securities collateral"],
        color="green",
        width=barWidth,
        label="own funds",
    )
    ax3.bar(
        banks_sorted,
        height=df_banks.loc[ix_sorted, "securities reused"],
        bottom=df_banks.loc[ix_sorted, "securities collateral"],
        color="red",
        width=barWidth,
        label="own funds",
    )
    ax3.legend(
        ["securities collateral", "securities reused"], loc="upper left"
    )
    ax3.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax3.set_title("Off Balance Sheets of Banks at step {}".format(int(step)))

    # plt.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    plt.savefig(
        path + "step_{}_balance_sheets.pdf".format(step),
        bbox_inches="tight",
    )
    plt.close()


def plot_assets_loans_mros(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["loans tot. network"],
    )
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["central bank funding tot. network"],
    )
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["total assets tot. network"],
    )
    # plt.plot(df_network_trajectory.index, df_network_trajectory["deposits"])
    plt.legend(["loans", "central bank funding", "total assets"])
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total assets, loans, and central bank funding")
    fig.tight_layout()
    plt.savefig(path + "Assets_loans_mros.pdf", bbox_inches="tight")
    plt.close()


def plot_network_density(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)

    for column in [
        f"network density-{agg_period}" for agg_period in par.agg_periods
    ]:
        plt.plot(
            df_network_trajectory.index,
            df_network_trajectory[column],
        )

    plt.xlabel("Steps")
    plt.ylabel("network density")
    plt.title("network density")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in par.agg_periods],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "network_density.pdf", bbox_inches="tight")
    plt.close()


def plot_gini(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(df_network_trajectory.index, df_network_trajectory["gini"])
    plt.xlabel("Steps")
    plt.ylabel("Gini coefficient")
    plt.title("Gini of banks' assets")
    fig.tight_layout()
    plt.savefig(path + "gini.pdf", bbox_inches="tight")
    plt.close()


def plot_reverse_repo_size_stats(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["repo exposures min network"],
    )
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["repo exposures max network"],
    )
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["repo exposures av. network"],
    )
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
    plt.savefig(path + "reverse_repo_stats.pdf", bbox_inches="tight")
    plt.close()


def plot_collateral_reuse(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index, df_network_trajectory["collateral reuse"]
    )
    plt.xlabel("Steps")
    plt.ylabel("Ratio")
    plt.title("collateral reuse")
    fig.tight_layout()
    plt.savefig(path + "collateral_reuse.pdf", bbox_inches="tight")
    plt.close()


def plot_repos(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["repo exposures tot. network"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total repo amount")
    fig.tight_layout()
    plt.savefig(path + "Repos_market_size.pdf", bbox_inches="tight")
    plt.close()


def plot_jaccard_not_aggregated(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)

    for column in [
        f"raw jaccard index-{agg_period}" for agg_period in par.agg_periods
    ]:
        plt.plot(
            df_network_trajectory.index,
            df_network_trajectory[column],
        )

    plt.xlabel("Steps")
    plt.ylabel("raw jaccard index")
    plt.title("raw jaccard index")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in par.agg_periods],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "raw_jaccard_index.pdf", bbox_inches="tight")
    plt.close()


def plot_jaccard_aggregated(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    for column in [
        f"jaccard index-{agg_period}" for agg_period in par.agg_periods
    ]:
        plt.plot(
            df_network_trajectory.index,
            df_network_trajectory[column],
        )
    plt.xlabel("Steps")
    plt.ylabel("jaccard index")
    plt.title("Jaccard index aggregated")
    plt.legend(
        [str(agg_period) + " time steps" for agg_period in par.agg_periods],
        loc="upper left",
    )
    plt.grid()
    fig.tight_layout()
    plt.savefig(path + "jaccard_index_agg.pdf", bbox_inches="tight")
    plt.close()


def plot_excess_liquidity_and_deposits(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["excess liquidity tot. network"],
    )
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["deposits tot. network"],
    )
    plt.legend(["excess liquidity" + "deposits"], loc="upper left")
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total excess liquidity & deposits")
    fig.tight_layout()
    plt.savefig(
        path + "excess_liquidity_and_deposits.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_collateral(df_network_trajectory, path):

    securities = [
        "securities usable",
        "securities encumbered",
    ] + par.off_bs_items
    keys = [f"{item} tot. network" for item in securities]

    fig = plt.figure(figsize=figsize)
    for key in keys:
        plt.plot(
            df_network_trajectory.index,
            df_network_trajectory[key],
        )

    plt.legend(
        securities,
        loc="upper left",
    )
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("Total collateral")
    fig.tight_layout()
    plt.savefig(path + "collateral.pdf", bbox_inches="tight")
    plt.close()


def plot_degre_network(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index, df_network_trajectory["av. in-degree"]
    )
    plt.xlabel("Steps")
    plt.ylabel("av. in-degree")
    plt.title("av. in-degree")
    fig.tight_layout()
    plt.savefig(path + "average_in-degree.pdf", bbox_inches="tight")
    plt.close()


def plot_average_nb_transactions(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["nb_ending_starting av. network"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Number of transactions")
    plt.title("nb_ending_starting av. network")
    fig.tight_layout()
    plt.savefig(
        path + "Average_nb_repo_transactions_ended.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_average_size_transactions(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["amount_ending_starting av. network"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Monetary units")
    plt.title("amount_ending_starting av. network")
    fig.tight_layout()
    plt.savefig(
        path + "Average_size_repo_transactions_ended.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_average_maturity_repo(df_network_trajectory, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_network_trajectory.index,
        df_network_trajectory["repos maturity av. network"],
    )
    plt.xlabel("Steps")
    plt.ylabel("Maturity")
    plt.title("repos maturity av. network")
    fig.tight_layout()
    plt.savefig(path + "Average_maturity_repo.pdf", bbox_inches="tight")
    plt.close()


def plot_network(adj, network_total_assets, path, step, name_in_title):
    # build a network from an adjacency matrix
    bank_network = nx.from_numpy_array(
        adj, parallel_edges=False, create_using=nx.DiGraph
    )
    # define the weight list from the weight information
    weights = [
        bank_network[node1][node2]["weight"]
        for node1, node2 in bank_network.edges()
    ]
    # log scale the big values in the repo network
    log_weights = [1 if i <= 1 else np.log(i) + 1 for i in weights]

    # define the size of the nodes a a function of the total deposits
    log_node_sizes = [
        0 if i <= 1 else np.log(i) + 1 for i in network_total_assets
    ]

    # define the position of the nodes
    pos = nx.spring_layout(bank_network)
    # pos = nx.circular_layout(bank_network)

    # draw the network
    fig = plt.figure(figsize=small_figsize)
    nx.draw_networkx(
        bank_network,
        pos,
        width=log_weights,
        with_labels=True,
        node_size=log_node_sizes,
    )

    # show the plot
    plt.title("{} network at the step {}".format(name_in_title, int(step)))
    fig.tight_layout()
    plt.savefig(
        path + "step_{}_network.pdf".format(step),
        bbox_inches="tight",
    )
    plt.close()


def plot_core_periphery(
    bank_network, sig_c, sig_x, path, step, name_in_title, figsize=(6, 3)
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


def plot_step_degree_per_asset_old(banks_tot_assets, banks_degree, path):
    fig = plt.figure(figsize=figsize)
    plt.scatter(banks_degree, banks_tot_assets)
    for i in range(len(banks_degree)):
        plt.text(x=banks_degree[i], y=banks_tot_assets[i], s=str(i))
    plt.ylabel("Degree")
    plt.xlabel("Assets")
    plt.title("Assets size verus degree")
    fig.tight_layout()
    plt.savefig(path + "Asset_per_degree.pdf", bbox_inches="tight")
    plt.close()


def plot_step_degree_per_asset(df_banks, agg_periods, path, figsize=(6, 3)):
    df_banks.plot(
        x="total assets",
        y=[f"degree_{agg_period}" for agg_period in agg_periods],
        figsize=figsize,
        style=".",
    )
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.ylabel("Degree")
    plt.xlabel("Assets")
    plt.xscale("log")
    plt.title("degree verus total assets")
    plt.savefig(path + "degree_per_asset.pdf", bbox_inches="tight")
    plt.close()


def plot_df_bank_trajectory(df_bank_trajectory, path):

    plt.figure(figsize=figsize)
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=slide_figsize)

    # Plot the accounting items (big)
    keys_s = [
        "loans",
        "deposits",
        "central bank funding",
    ]

    for key in keys_s:
        ax1.plot(
            df_bank_trajectory.index, df_bank_trajectory[key], label=str(key)
        )

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
        "cash",
        "securities usable",
        "securities encumbered",
        # "reverse repo exposures",
        # "own funds",
        # "repo exposures",
        "securities collateral",
        "securities reused",
    ]

    for key in keys_s:
        ax2.plot(
            df_bank_trajectory.index, df_bank_trajectory[key], label=str(key)
        )

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
        "av. in-degree",
        "av. out-degree",
        "nb_ending_starting",
        "repos maturity av. bank",
    ]
    for key in keys_s:
        ax3.plot(
            df_bank_trajectory.index, df_bank_trajectory[key], label=str(key)
        )

    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Indicators")
    ax3.legend(loc="upper left")
    ax3.grid()
    ax3.set_title("Single bank trajectory of indicators")

    fig.tight_layout()
    plt.savefig(
        path + "Single_trajectory_indicators.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_multiple_key(
    param_values, input_param, output, agg_periods, path, short_key, nb_char
):
    fig = plt.figure(figsize=figsize)
    for key in output.keys():
        if key[0:nb_char] == short_key:
            plt.plot(param_values, output[key], "-o")
    if input_param in par.log_input_params:
        plt.gca().set_xscale("log")
        plt.xlabel(input_param + " (log-scale)")
    else:
        plt.xlabel(input_param)

    if (
        short_key in par.log_output_mlt_keys
        and input_param in par.log_input_params
    ):
        plt.ylabel(short_key + " (log-scale)")
        plt.gca().set_yscale("log")
    else:
        plt.ylabel(short_key)

    plt.legend(
        [str(agg_period) + " time steps" for agg_period in agg_periods],
        loc="upper left",
    )
    plt.title(short_key + "x periods as a fct. of " + input_param)
    fig.tight_layout()
    plt.savefig(
        path + short_key + "_agg_" + input_param + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_single_key(key, param_values, input_param, output, path):
    fig = plt.figure(figsize=figsize)
    plt.plot(param_values, output[key], "-o")
    if input_param in par.log_input_params:
        plt.gca().set_xscale("log")
        plt.xlabel(input_param + " (log-scale)")
    else:
        plt.xlabel(input_param)
    if (
        key in par.log_output_single_keys
        and input_param in par.log_input_params
    ):
        plt.ylabel(key + " (log-scale)")
        plt.gca().set_yscale("log")
    else:
        plt.ylabel(key)
    plt.title(key + " as a fct. of " + input_param)
    fig.tight_layout()
    plt.savefig(path + key + "_" + input_param + ".pdf", bbox_inches="tight")
    plt.close()


def plot_output_by_param(
    param_values, input_param, output, jaccard_periods, agg_periods, path
):
    output = fct.reformat_output(output)

    # plot multiple keys on the same chart
    for short_key in par.output_mlt_keys:
        plot_multiple_key(
            param_values=param_values,
            input_param=input_param,
            output=output,
            agg_periods=agg_periods,
            path=path,
            short_key=short_key,
            nb_char=len(short_key),
        )

    # case of the jaccard old version (with jaccard perriods != agg periodes)
    plot_multiple_key(
        param_values=param_values,
        input_param=input_param,
        output=output,
        agg_periods=jaccard_periods,
        path=path,
        short_key="jaccard index",
        nb_char=len(short_key),
    )

    # plot all other output metrics
    for key in par.output_single_keys:
        plot_single_key(
            key=key,
            param_values=param_values,
            input_param=input_param,
            output=output,
            path=path,
        )
