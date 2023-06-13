import os
import numpy as np
import cpnet
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import functions as fct
import parameters as par
import emp_metrics as em
from tqdm import tqdm


def plot(Dynamics):

    # create paths
    path_results = Dynamics.path_results
    os.makedirs(f"{path_results}accounting_view/static/", exist_ok=True)
    os.makedirs(f"{path_results}transaction_view/", exist_ok=True)
    os.makedirs(f"{path_results}exposure_view/core-periphery/", exist_ok=True)

    plot_all_network_trajectory(Dynamics)
    plot_all_dashed_trajectory(Dynamics)

    # Plot the single bank trajectory time series.
    plot_bank_trajectory(
        Dynamics.df_bank_trajectory,
        f"{path_results}exposure_view/",
    )


def plot_all_network_trajectory(Dynamics):
    for index in par.df_figures.index:
        plot_network_trajectory(
            df=Dynamics.df_network_trajectory,
            cols=[
                f"{item}{par.df_figures.loc[index,'extension']}"
                for item in par.df_figures.loc[index, "items"]
            ],
            file_name=f"{Dynamics.path_results}{index}.pdf",
        )


def plot_network_trajectory(
    df,
    cols,
    file_name,
    xscale=False,
    figsize=par.small_figsize,
):
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("flare", n_colors=len(cols))

    # convert the cols of the df using the convertion of first col
    df = convert_data(df[cols], par.df_plt.loc[cols[0], "convertion"])

    for i, col in enumerate(cols):
        try:
            plt.plot(
                df.index,
                df[col],
                par.df_plt.loc[col, "style"],
                color=colors[i],
            )

        except:
            print(f"bug for {col}")

    plt.legend(
        par.df_plt.loc[cols, "legend"],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    plt.xlabel("time (days)")
    plt.ylabel(par.df_plt.loc[cols[0], "label"])
    if xscale:
        plt.xscale(xscale)
    plt.yscale(par.df_plt.loc[cols[0], "scale"])
    plt.grid()
    plt.savefig(f"{file_name}", bbox_inches="tight")
    plt.close()


def plot_all_dashed_trajectory(Dynamics):

    path_results = Dynamics.path_results
    plot_period = Dynamics.plot_period
    days = range(Dynamics.Network.step + 1)  # to cover all steps up to now
    dic_dashed_trajectory = Dynamics.dic_dashed_trajectory

    # --------------
    # exposure view

    # plot the reverse repo network
    plot_weighted_adj_network(
        Dynamics.arr_rev_repo_exp_adj,
        dic_dashed_trajectory,
        days,
        plot_period,
        f"{path_results}exposure_view/weighted_adj_network/",
        "reverse repo",
    )

    # plot degree distribution
    plot_degree_distribution(
        Dynamics.dic_in_degree,
        Dynamics.dic_out_degree,
        days,
        plot_period,
        f"{path_results}exposure_view/degree_distribution/",
    )

    # Plot degree per asset
    plot_degree_per_asset(
        dic_dashed_trajectory,
        Dynamics.dic_degree,
        range(Dynamics.Network.nb_banks),
        days,
        plot_period,
        f"{path_results}exposure_view/degree_per_asset/",
    )

    # Plot the core-periphery struture (warning heavy plot : app. 3000 plots per run)
    if Dynamics.cp_option and Dynamics.heavy_plot:
        plot_cpnet(
            df_network_trajectory=Dynamics.df_network_trajectory,
            dic_arr_binary_adj=Dynamics.dic_arr_binary_adj,
            arr_rev_repo_exp_adj=Dynamics.arr_rev_repo_exp_adj,
            days=days,
            plot_period=Dynamics.plot_period,
            path=f"{path_results}exposure_view/",
        )

    # --------------
    # accounting view

    # Plot the break-down of the balance per bank
    plot_balance_sheet(
        dic_dashed_trajectory,
        days,
        plot_period,
        f"{path_results}accounting_view/balance_sheet/",
    )

    # plot powerlaws (warning heavy plot : app. 500 plots per run)
    if Dynamics.heavy_plot:
        plot_powerlaw(
            df_network_trajectory=Dynamics.df_network_trajectory,
            days=days,
            plot_period=plot_period,
            path=f"{path_results}accounting_view/",
        )


def plot_step_weighted_adj_network(
    adj,
    df_banks,
    days,
    step,
    path,
    name,
    figsize=par.small_figsize,
    bank_ids=False,
    log_node=False,
):
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

    # filter df_banks to keep only bank_ids
    if bank_ids:
        finrep_bank_ids = df_banks.index
        df_banks = df_banks.loc[
            fct.list_intersection(finrep_bank_ids, bank_ids)
        ]

    # define the size of the nodes a a function of the total deposits
    if log_node:
        node_sizes = [
            0 if amount <= 1 else np.log(amount) + 1
            for amount in df_banks["total assets"]
        ]
    else:
        node_sizes = df_banks["total assets"]

    # define the position of the nodes
    pos = nx.spring_layout(bank_network)

    # draw the network
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        bank_network,
        pos,
        width=log_weights,
        with_labels=True,
        node_size=node_sizes,
    )

    # show the plot
    if isinstance(days[step], pd.Timestamp):
        day_print = days[step].strftime("%Y-%m-%d")
    else:
        day_print = days[step]
    plt.title(f"{name} network on day {day_print}")
    plt.savefig(
        f"{path}weighted_adj_network_on_day_{day_print}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_weighted_adj_network(
    arr_adj,
    dic_dashed_trajectory,
    days,
    plot_period,
    path,
    name,
    figsize=par.small_figsize,
    bank_ids=False,
    plot_days=False,
):

    os.makedirs(path, exist_ok=True)

    if plot_days:
        plot_steps = fct.get_plot_steps_from_days(days, plot_days)

    else:
        plot_steps = fct.get_plot_steps_from_period(days, plot_period)

    for step in plot_steps:
        plot_step_weighted_adj_network(
            arr_adj[step],
            dic_dashed_trajectory[days[step]],
            days,
            step,
            path,
            name=name,
            figsize=figsize,
            bank_ids=bank_ids,
        )


def plot_step_degree_distribution(
    dic_in_degree,
    dic_out_degree,
    days,
    step,
    path,
    figsize=par.small_figsize,
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    width = 0.1
    space = 0.05
    pos = 0

    colors = sns.color_palette("flare", n_colors=len(par.agg_periods))

    for i, agg_period in enumerate(par.agg_periods):

        # in degree plot
        hist = np.unique(
            dic_in_degree[agg_period][step],
            return_counts=True,
        )
        ax1.bar(hist[0] + pos, hist[1], width=width, color=colors[i])

        # out degree plot
        hist = np.unique(
            dic_out_degree[agg_period][step],
            return_counts=True,
        )
        ax2.bar(hist[0] + pos, hist[1], width=width, color=colors[i])
        pos = pos + width + space

    ax1.set_xlabel("degree")
    ax1.set_ylabel("frequency")
    ax1.set_title("distribution of in-degree")
    ax1.legend(
        [str(agg_period) + " time steps" for agg_period in par.agg_periods],
        loc="upper left",
    )

    ax2.set_xlabel("degree")
    ax2.set_title("distribution of out-degree")
    ax2.legend(
        [str(agg_period) + " time steps" for agg_period in par.agg_periods],
        loc="upper left",
    )

    if isinstance(days[step], pd.Timestamp):
        day_print = days[step].strftime("%Y-%m-%d")
    else:
        day_print = days[step]
    plt.savefig(
        f"{path}degree_distribution_on_day_{day_print}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_degree_distribution(
    dic_in_degree,
    dic_out_degree,
    days,
    plot_period,
    path,
    figsize=par.small_figsize,
):

    os.makedirs(path, exist_ok=True)

    plot_steps = fct.get_plot_steps_from_period(days, plot_period)

    for step in plot_steps:
        plot_step_degree_distribution(
            dic_in_degree,
            dic_out_degree,
            days,
            step,
            path,
            figsize=figsize,
        )


def plot_step_degree_per_asset(
    df_banks,
    dic_degree,
    bank_ids,
    days,
    step,
    path,
    figsize=(6, 3),
    finrep_bank_ids=False,
):

    # build the degree per bank on the date step
    df_degree = pd.DataFrame(index=bank_ids)
    for agg_period in par.agg_periods:
        df_degree[f"degree-{agg_period}"] = dic_degree[agg_period][step]

    if not (finrep_bank_ids):
        finrep_bank_ids = bank_ids

    # merge the 2 df
    df = pd.merge(
        df_banks[["total assets"]],
        df_degree,
        right_index=True,
        left_index=True,
        how="inner",
    )

    df.plot(
        x="total assets",
        y=[f"degree-{agg_period}" for agg_period in par.agg_periods],
        figsize=figsize,
        style=".",
        colormap=ListedColormap(
            sns.color_palette("flare", n_colors=len(par.agg_periods))
        ),
    )

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.ylabel("Degree")
    plt.xlabel("Assets")
    plt.xscale("log")

    if isinstance(days[step], pd.Timestamp):
        day_print = days[step].strftime("%Y-%m-%d")
    else:
        day_print = days[step]

    plt.title(f"degree verus total assets on day {day_print}")
    plt.savefig(
        f"{path}degree_per_asset_on_day_{day_print}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_degree_per_asset(
    dic_dashed_trajectory,
    dic_degree,
    bank_ids,
    days,
    plot_period,
    path,
    figsize=(6, 3),
    plot_days=False,
    finrep_bank_ids=False,
):

    os.makedirs(path, exist_ok=True)

    if plot_days:
        plot_steps = fct.get_plot_steps_from_days(days, plot_days)

    else:
        plot_steps = fct.get_plot_steps_from_period(days, plot_period)

    for step in plot_steps:
        plot_step_degree_per_asset(
            dic_dashed_trajectory[days[step]],
            dic_degree,
            bank_ids,
            days,
            step,
            path,
            figsize=figsize,
            finrep_bank_ids=finrep_bank_ids,
        )


def plot_cpnet(
    df_network_trajectory,
    dic_arr_binary_adj,
    arr_rev_repo_exp_adj,
    days,
    plot_period,
    path,
    figsize=par.small_figsize,
):

    plot_steps = fct.get_plot_steps_from_period(days, plot_period)

    # loop across agg periods
    for agg_period in par.agg_periods + ["weighted"]:

        # define the arr_adj to be used
        if agg_period == "weighted":
            arr_adj = arr_rev_repo_exp_adj
        else:
            arr_adj = dic_arr_binary_adj[agg_period]

        # define df_pvalue
        df_cpnet_pvalue = pd.DataFrame(columns=par.cp_algos)

        # loop across algos
        for algo in par.cp_algos:

            # loop across plot step
            for step in plot_steps:
                day = days[step]

                # plot a cpnet structure (commented as it generates 3000 nodes per run)
                # retrieve the cp structure from df_network trajectory
                sig_c = df_network_trajectory.loc[
                    day, f"cpnet sig_c {algo}-{agg_period}"
                ]
                sig_x = df_network_trajectory.loc[
                    day, f"cpnet sig_x {algo}-{agg_period}"
                ]
                plot_step_algo_cpnet(
                    sig_c,
                    sig_x,
                    f"{path}core-periphery/{agg_period}/{algo}/",
                    day,
                    arr_adj[step],
                    figsize,
                )


def plot_step_algo_cpnet(sig_c, sig_x, path, day, adj, figsize):
    # check that there is a result to plot
    if sig_c != {}:

        # define path
        os.makedirs(path, exist_ok=True)

        # build nx object
        bank_network = nx.from_numpy_array(
            adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )

        # plot cp structure
        fig, ax = plt.subplots(figsize=figsize)
        ax = plt.gca()
        ax, pos = cpnet.draw(bank_network, sig_c, sig_x, ax)
        if isinstance(day, pd.Timestamp):
            day_print = day.strftime("%Y-%m-%d")
        else:
            day_print = day
        plt.savefig(
            f"{path}core-periphery_structure_on_day_{day_print}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_powerlaw(
    df_network_trajectory,
    days,
    plot_period,
    path,
    figsize=par.small_figsize,
    plot_days=False,
    bank_items=False,
):
    if not (plot_days):
        plot_days = fct.get_plot_days_from_period(days, plot_period)

    if not(bank_items):
        bank_items = par.bank_items

    for day in plot_days:

        if isinstance(day, pd.Timestamp):
            day_print = day.strftime("%Y-%m-%d")
        else:
            day_print = day

        for bank_item in bank_items:

            # retrive the information to plot
            powerlaw_fit = df_network_trajectory.loc[
                day, f"powerlaw fit {bank_item}"
            ]
            powerlaw_alpha = df_network_trajectory.loc[
                day, f"powerlaw alpha {bank_item}"
            ]
            powerlaw_pvalue = df_network_trajectory.loc[
                day, f"powerlaw p-value {bank_item}"
            ]

            # plot
            if not (np.isnan(powerlaw_pvalue)):
                plot_step_item_powerlaw(
                    powerlaw_fit,
                    powerlaw_alpha,
                    powerlaw_pvalue,
                    bank_item,
                    f"{path}static/on_day_{day_print}/",
                    figsize=figsize,
                )


def plot_step_item_powerlaw(
    powerlaw_fit,
    powerlaw_alpha,
    powerlaw_pvalue,
    bank_item,
    path,
    figsize=par.small_figsize,
    auto_xlabel=False,
):

    # define the figure and colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = sns.color_palette("flare", n_colors=3)

    # ax1 : pdf
    try:
        powerlaw_fit.plot_pdf(color=colors[0], ax=ax1)

    except:
        print(f"bug for {bank_item}")

    powerlaw_fit.power_law.plot_pdf(color=colors[1], linestyle="--", ax=ax1)
    powerlaw_fit.exponential.plot_pdf(color=colors[2], linestyle="--", ax=ax1)
    if auto_xlabel:
        ax1.set_xlabel(par.df_plt.loc[bank_item, "legend"])
    ax1.set_ylabel("pdf")
    ax1.grid()

    # ax2 : ccdf
    powerlaw_fit.plot_ccdf(color=colors[0], ax=ax2)
    powerlaw_fit.power_law.plot_ccdf(color=colors[1], linestyle="--", ax=ax2)
    powerlaw_fit.exponential.plot_ccdf(color=colors[2], linestyle="--", ax=ax2)
    if auto_xlabel:
        ax2.set_xlabel(par.df_plt.loc[bank_item, "legend"])
    ax2.set_ylabel("ccdf")
    ax2.grid()

    # legend
    powerlaw_alpha_fm = "{:.1f}".format(powerlaw_alpha)
    powerlaw_pvalue_fm = "{:.1f}".format(powerlaw_pvalue)
    ax2.legend(
        [
            "data",
            r"power law fit, $\alpha =$"
            + f"{powerlaw_alpha_fm}, p-value = {powerlaw_pvalue_fm}",
            "exponential",
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )

    # adjust the space between the 2 charts
    plt.subplots_adjust(wspace=0.4)
    os.makedirs(path, exist_ok=True)
    try:
        plt.savefig(f"{path}{bank_item}.pdf", bbox_inches="tight")
    except:
        print(f"bug for {bank_item}")
    plt.close()


def plot_balance_sheet(
    dic_dashed_trajectory,
    days,
    plot_period,
    path,
):

    os.makedirs(path, exist_ok=True)

    plot_days = fct.get_plot_days_from_period(days, plot_period)

    for day in plot_days:
        df_banks = dic_dashed_trajectory[day]
        plot_step_balance_sheet(df_banks, day, path)


def plot_step_balance_sheet(
    df_banks,
    day,
    path,
):

    if isinstance(day, pd.Timestamp):
        day_print = day.strftime("%Y-%m-%d")
    else:
        day_print = day

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=par.halfslide_figsize)

    ix_sorted = np.argsort(np.asarray(df_banks["total assets"]))
    banks_sorted = ["bank {}".format(str(b)) for b in ix_sorted]

    asset_bottoms = [np.zeros(len(banks_sorted))]
    for col in par.assets[:-1]:
        asset_bottoms.append(asset_bottoms[-1] + df_banks.loc[ix_sorted, col])
    liability_bottoms = [np.zeros(len(banks_sorted))]
    for col in par.liabilities[:-1]:
        liability_bottoms.append(
            liability_bottoms[-1] + df_banks.loc[ix_sorted, col]
        )
    off_bottoms = [np.zeros(len(banks_sorted))]
    for col in par.off_bs_items[:-1]:
        off_bottoms.append(off_bottoms[-1] + df_banks.loc[ix_sorted, col])

    barWidth = 0.75

    asset_colors = sns.color_palette("flare", n_colors=len(par.assets))
    liabilities_colors = sns.color_palette(
        "crest", n_colors=len(par.liabilities)
    )
    off_colors = sns.color_palette("Blues", n_colors=len(par.off_bs_items))

    for i, col in enumerate(par.assets):
        ax1.bar(
            banks_sorted,
            height=df_banks.loc[ix_sorted, col],
            bottom=asset_bottoms[i],
            color=asset_colors[i],
            width=barWidth,
            label=col,
        )
    ax1.legend(par.assets)
    ax1.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax1.set_title(f"Assets on day {day_print}")

    for i, col in enumerate(par.liabilities):
        ax2.bar(
            banks_sorted,
            height=df_banks.loc[ix_sorted, col],
            bottom=liability_bottoms[i],
            color=liabilities_colors[i],
            width=barWidth,
            label=col,
        )
    ax2.legend(par.liabilities)
    ax2.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax2.set_title(f"Liabilities on day {day_print}")

    for i, col in enumerate(par.off_bs_items):
        ax3.bar(
            banks_sorted,
            height=df_banks.loc[ix_sorted, col],
            bottom=off_bottoms[i],
            color=off_colors[i],
            width=barWidth,
            label=col,
        )
    ax3.legend(par.off_bs_items)
    ax3.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax3.set_title(f"Off-balance sheet items on day {day_print}")

    # plt.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    plt.savefig(
        f"{path}balance_sheets_on_day_{day_print}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_bank_trajectory(df_bank_trajectory, path, figsize=par.slide_figsize):

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=figsize)

    # Plot the accounting items (big)
    keys = [
        "loans",
        "deposits",
        "central bank funding",
    ]

    for key in keys:
        ax1.plot(
            df_bank_trajectory.index,
            df_bank_trajectory[key],
            label=str(key),
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
    keys = [
        "cash",
        "securities usable",
        "securities encumbered",
        # "reverse repo balance",
        # "own funds",
        # "repo balance",
        "securities collateral",
        "securities reused",
    ]

    for key in keys:
        ax2.plot(
            df_bank_trajectory.index,
            df_bank_trajectory[key],
            label=str(key),
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
    in_colors = sns.color_palette("flare", n_colors=len(par.agg_periods))
    out_colors = sns.color_palette("crest", n_colors=len(par.agg_periods))
    for i, agg_period in enumerate(par.agg_periods):
        ax3.plot(
            df_bank_trajectory.index,
            df_bank_trajectory[f"in-degree-{agg_period}"],
            label=f"in-degree-{agg_period}",
            color=in_colors[i],
        )
    for i, agg_period in enumerate(par.agg_periods):
        ax3.plot(
            df_bank_trajectory.index,
            df_bank_trajectory[f"out-degree-{agg_period}"],
            label=f"out-degree-{agg_period}",
            color=out_colors[i],
        )

    keys = [
        "number repo transactions av. bank",
        "repo transactions maturity av. bank",
    ]
    for key in keys:
        ax3.plot(
            df_bank_trajectory.index,
            df_bank_trajectory[key],
            label=str(key),
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


def plot_sensitivity(
    df_network_sensitivity,
    input_parameter,
    cols,
    file_name,
    figsize=par.small_figsize,
):

    # define figure
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("flare", n_colors=len(cols))

    # filter on index
    df = df_network_sensitivity.loc[input_parameter].copy()

    # convert the cols of the df using the convertion of first col
    df = convert_data(df[cols], par.df_plt.loc[cols[0], "convertion"])

    # convert the index
    df = convert_index(
        df,
        par.df_plt.loc[input_parameter, "convertion"],
    )

    # sort index
    df.sort_index(inplace=True)

    # filter on columns
    for i, col in enumerate(cols):
        plt.plot(df.index, df[col], ".-", color=colors[i])

    # set legend
    lgd = plt.legend(
        par.df_plt.loc[cols, "legend"], loc="upper left", bbox_to_anchor=(1, 1)
    )
    plt.xlabel(par.df_plt.loc[input_parameter, "label"])
    plt.ylabel(par.df_plt.loc[cols[0], "label"])
    plt.xscale(par.df_plt.loc[input_parameter, "scale"])
    plt.yscale(par.df_plt.loc[cols[0], "scale"])
    plt.grid()
    plt.savefig(
        f"{file_name}",
        bbox_inches="tight",
        bbox_extra_artists=(lgd,),
    )
    plt.close()


def plot_all_sensitivities(df_network_sensitivity, path):

    # loop over the metrics
    for metric in tqdm(par.df_figures.index):

        # loop over the input parameters
        for input_parameter in pd.unique(
            df_network_sensitivity.index.get_level_values(0)
        ):

            # create path
            os.makedirs(f"{path}/{metric}/", exist_ok=True)

            # plot
            plot_sensitivity(
                df_network_sensitivity=df_network_sensitivity,
                input_parameter=input_parameter,
                cols=[
                    f"{item}{par.df_figures.loc[metric,'extension']}"
                    for item in par.df_figures.loc[metric, "items"]
                ],
                file_name=f"{path}{metric}/{input_parameter}.pdf",
            )


def convert_data(df, str_convertion):

    if str_convertion == "e-K$":
        df = np.exp(df) / 1e3
    elif str_convertion == "e":
        df = np.exp(df)
    elif str_convertion == "K$":
        df = df / 1e3
    elif str_convertion == "%":
        df = df * 100

    return df


def convert_index(df, str_convertion):

    index = df.index

    if str_convertion == "e-K$":
        index = np.exp(index) / 1e3
    elif str_convertion == "e":
        index = np.exp(index)
    elif str_convertion == "K$":
        index = index / 1e3
    elif str_convertion == "%":
        index = index * 100

    df.index = pd.Index(index)

    return df


def plot_notional_by_notice_period(
    df_mmsr_secured, path, plot_period, figsize=par.small_figsize
):

    # select only the evergreen, with repeated lines by day, tenor is the notice period
    df = df_mmsr_secured[df_mmsr_secured["evergreen"]]

    # select the dates on which to plot
    days = pd.to_datetime(
        sorted(
            list(set(df_mmsr_secured["trade_date"].dt.strftime("%Y-%m-%d")))
        )
    )
    plot_days = fct.get_plot_days_from_period(days, plot_period)

    # initialize path
    os.makedirs(f"{path}notional_by_notice_period/", exist_ok=True)

    # looop over the plot days
    for day in plot_days:

        # build notional by notice period
        df_notional_by_notice_period = (
            df[df["trade_date"] == day]
            .groupby(["tenor"])
            .agg({"trns_nominal_amt": sum})
        )

        df_notional_by_notice_period.plot(
            figsize=figsize,
            legend=False,
            colormap=ListedColormap(
                sns.color_palette("flare", n_colors=len(par.agg_periods))
            ),
        )
        plt.xlabel("notice period (days)")
        plt.xscale("log")
        plt.ylabel("notional (monetary units)")
        plt.grid()

        # define day print
        day_print = day.strftime("%Y-%m-%d")

        # save file
        df_notional_by_notice_period.to_csv(
            f"{path}notional_by_notice_period/df_notional_by_notice_period_on_day_{day_print}.csv"
        )

        # save fig
        plt.savefig(
            f"{path}notional_by_notice_period/notional_by_notice_period_on_day_{day_print}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_collateral_reuse(df_isin, path, plot_period, figsize=par.small_figsize):

    days = df_isin.index.get_level_values("current_date").unique()
    plot_days = fct.get_plot_days_from_period(days, plot_period)

    # do a for loop here
    day = plot_days[2]

    for day in plot_days:
        df_isin.loc[day, "trns_nominal_amt"].hist(
            figsize=figsize,
            legend=False,
            color=sns.color_palette("flare", n_colors=1),
        )

        plt.xlabel("collateral reuse (#)")
        plt.ylabel("frequency")
        plt.grid()

        # define day print
        day_print = day.strftime("%Y-%m-%d")

        # save fig
        plt.savefig(
            f"{path}collateral_reuse/collateral_reuse_on_day_{day_print}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_dyn_powerlaw_tranverse(df_powerlaw,path, figsize=par.small_figsize):

    bank_ids = [ind.split(" ")[0] for ind in df_powerlaw.index]
    extensions = [" abs. var.",  "rel. var.", " var over tot. assets"]
    indexes = [fct.list_intersection([f"{bank_id}{extension}" for bank_id in bank_ids],list(df_powerlaw.index)) for extension in extensions]

    for col in df_powerlaw.columns:

        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("flare", n_colors=len(extensions))

        for i, (index, ext) in enumerate(zip(indexes,extensions)):
            plt.plot(
                [ind.split(" ")[0] for ind in index],
                df_powerlaw.loc[index,col],
                color=colors[i],
            )

        plt.xlabel("bank ids")
        plt.ylabel(f"{col}")
        plt.grid()

        # save fig
        plt.savefig(
            f"{path}dyn_powerlaw_{col}.pdf",
            bbox_inches="tight",
        )
        plt.close()
