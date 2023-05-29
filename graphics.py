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
import powerlaw
import sys


def plot(Dynamics):

    # create paths
    path_results = Dynamics.path_results
    os.makedirs(f"{path_results}accounting_view/", exist_ok=True)
    os.makedirs(f"{path_results}transaction_view/", exist_ok=True)
    os.makedirs(f"{path_results}exposure_view/core-periphery", exist_ok=True)

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
    xscale="linear",
    figsize=par.small_figsize,
):
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("flare", n_colors=len(cols))

    # convert the cols of the df using the convertion of first col
    df = convert_data(df[cols], par.df_plt.loc[cols[0], "convertion"])

    for i, col in enumerate(cols):
        plt.plot(
            df.index,
            df[col],
            par.df_plt.loc[col, "style"],
            color=colors[i],
        )
    plt.legend(
        par.df_plt.loc[cols, "legend"],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    plt.xlabel("time (days)")
    plt.ylabel(par.df_plt.loc[cols[0], "label"])
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
        Dynamics.arr_total_assets,
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
        Dynamics.arr_total_assets,
        Dynamics.dic_degree,
        range(Dynamics.Network.nb_banks),
        days,
        plot_period,
        f"{path_results}exposure_view/degree_per_asset/",
    )

    # # commented because it creates too many figures (costs up to 3000 nodes)
    # # Plot the core-periphery detection and assessment
    # if Dynamics.cp_option:
    #     plot_cpnet(
    #         df_network_trajectory=Dynamics.df_network_trajectory,
    #         dic_arr_binary_adj=Dynamics.dic_arr_binary_adj,
    #         arr_rev_repo_exp_adj=Dynamics.arr_rev_repo_exp_adj,
    #         days=days,
    #         plot_period=Dynamics.plot_period,
    #         path=f"{path_results}exposure_view/",
    #     )

    # --------------
    # accounting view

    # Plot the break-down of the balance per bank
    plot_balance_sheet(
        dic_dashed_trajectory,
        days,
        plot_period,
        f"{path_results}accounting_view/balance_sheet/",
    )

    plot_all_power_law(
        dic_dashed_trajectory,
        days,
        plot_period,
        f"{path_results}accounting_view/power_laws/",
    )


def plot_step_weighted_adj_network(
    adj,
    ar_total_assets,
    days,
    step,
    path,
    name,
    figsize=par.small_figsize,
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

    # define the size of the nodes a a function of the total deposits
    log_node_sizes = [0 if i <= 1 else np.log(i) + 1 for i in ar_total_assets]

    # define the position of the nodes
    pos = nx.spring_layout(bank_network)

    # draw the network
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        bank_network,
        pos,
        width=log_weights,
        with_labels=True,
        node_size=log_node_sizes,
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
    arr_total_assets,
    days,
    plot_period,
    path,
    name,
    figsize=par.small_figsize,
):

    os.makedirs(path, exist_ok=True)

    plot_steps = fct.get_plot_steps_from_period(days, plot_period)

    for step in plot_steps:
        plot_step_weighted_adj_network(
            arr_adj[step],
            arr_total_assets[step],
            days,
            step,
            path,
            name=name,
            figsize=figsize,
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
    ar_total_assets,
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

    # build the asset per bank from the ar_total assets
    df_total_assets = pd.DataFrame(
        ar_total_assets, index=finrep_bank_ids, columns=["total assets"]
    )

    # merge the 2 df
    df = pd.merge(
        df_total_assets,
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

    plt.title(f"degree verus total assets at step {day_print}")
    plt.savefig(
        f"{path}degree_per_asset_on_day_{day_print}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_degree_per_asset(
    arr_total_assets,
    dic_degree,
    bank_ids,
    days,
    plot_period,
    path,
    figsize=(6, 3),
    plot_days=False,
    finrep_days=False,
    finrep_bank_ids=False,
):

    os.makedirs(path, exist_ok=True)

    if plot_days:
        plot_steps = fct.get_plot_steps_from_days(days, plot_days)
        finrep_plot_steps = fct.get_plot_steps_from_days(
            finrep_days, plot_days
        )

    else:
        plot_steps = fct.get_plot_steps_from_period(days, plot_period)
        finrep_plot_steps = plot_steps

    for step, finrep_step in zip(plot_steps, finrep_plot_steps):
        plot_step_degree_per_asset(
            arr_total_assets[finrep_step],
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
                plot_step_cpnet(
                    sig_c,
                    sig_x,
                    f"{path}core-periphery/{agg_period}/{algo}/",
                    day,
                    arr_adj[step],
                    figsize,
                )


def plot_step_cpnet(sig_c, sig_x, path, day, adj, figsize):
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
            f"{path}core-periphery_structure_step_{day_print}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_all_power_law(
    dic_dashed_trajectory,
    days,
    plot_period,
    path,
    figsize=par.small_figsize,
):

    os.makedirs(path, exist_ok=True)

    plot_days = fct.get_plot_days_from_period(days, plot_period)

    for day in plot_days:
        df_banks = dic_dashed_trajectory[day]
        plot_step_all_power_law(df_banks, day, path, figsize=figsize)


def plot_step_all_power_law(df_banks, day, path, figsize=par.small_figsize):

    # replace all values smaller or equal to 0 by nan (to avoid powerlaw warnings)
    df = df_banks.mask(df_banks <= 0)

    # build path
    if isinstance(day, pd.Timestamp):
        day_print = day.strftime("%Y-%m-%d")
    else:
        day_print = day
    path_pwl = f"{path}{day_print}/"
    os.makedirs(f"{path_pwl}", exist_ok=True)

    df_power_law = pd.DataFrame(
        index=df.columns, columns=["alpha_power_law", "p_value"]
    )

    for col in df.columns:

        # define data
        sr_data = df[col]

        # check that some data exists
        if not (sr_data.isna().all()):

            # run and plot the power law test
            df_power_law.loc[col] = plot_step_power_law(
                sr_data, f"{path_pwl}{col}", figsize=figsize
            )

    # dump to csv
    df_power_law.to_csv(f"{path}df_power_law.csv")

    # plot
    fig, ax1 = plt.subplots(figsize=figsize)
    colors = sns.color_palette("flare", n_colors=2)

    # ax1: alpha
    ax1.plot(
        df_power_law.index,
        df_power_law["alpha_power_law"],
        ".",
        color=colors[0],
    )
    ax1.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax1.set_xlabel("accounting item")
    ax1.set_ylabel(r"$\alpha$")
    ax1.legend([r"$\alpha$"], loc="upper left")

    # ax2: p value
    ax2 = ax1.twinx()
    ax2.plot(df_power_law.index, df_power_law["p_value"], ".", color=colors[1])
    ax2.set_ylabel("p-value")
    ax2.legend(["p-value"], loc="upper right")

    plt.grid()
    plt.savefig(
        f"{path}power_law_tests_on_day_{day_print}.pdf", bbox_inches="tight"
    )
    plt.close()


def plot_step_power_law(sr_data, file_name, figsize=par.small_figsize):

    # at least 2 data points required with non negligeable size
    if (len(sr_data.dropna()) > 1) and (sr_data.abs().sum() > par.float_limit):

        # fit the data with the powerlaw librairy
        fit = powerlaw.Fit(sr_data.dropna())

        # define the figure and colors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        colors = sns.color_palette("flare", n_colors=3)
        col = sr_data.name

        # ax1 : pdf
        fit.plot_pdf(color=colors[0], ax=ax1)
        fit.power_law.plot_pdf(color=colors[1], linestyle="--", ax=ax1)
        fit.exponential.plot_pdf(color=colors[2], linestyle="--", ax=ax1)
        ax1.set_xlabel(par.df_plt.loc[col, "legend"])
        ax1.set_ylabel("pdf")
        ax1.grid()

        # ax2 : ccdf
        fit.plot_ccdf(color=colors[0], ax=ax2)
        fit.power_law.plot_ccdf(color=colors[1], linestyle="--", ax=ax2)
        fit.exponential.plot_ccdf(color=colors[2], linestyle="--", ax=ax2)
        ax2.set_xlabel(par.df_plt.loc[col, "legend"])
        ax2.set_ylabel("ccdf")
        ax2.grid()

        # legend
        alpha_power_law = fit.power_law.alpha
        alpha_power_law_fm = "{:.1f}".format(alpha_power_law)
        R, p_value = fit.distribution_compare("power_law", "exponential")
        p_value_fm = "{:.1f}".format(p_value)
        ax2.legend(
            [
                "data",
                r"power law fit, $\alpha =$"
                + f"{alpha_power_law_fm}, p-value = {p_value_fm}",
                "exponential",
            ],
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
        )

        # adjust the space between the 2 charts
        plt.subplots_adjust(wspace=0.4)

        plt.savefig(f"{file_name}.pdf", bbox_inches="tight")
        plt.close()

    else:
        alpha_power_law = np.nan
        p_value = np.nan

    return alpha_power_law, p_value


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
    df = convert_n_format_index(
        df,
        par.df_plt.loc[input_parameter, "format"],
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
    for metric in par.df_figures.index:

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


def convert_n_format_index(df, str_format, str_convertion):

    index = df.index

    if str_convertion == "e-K$":
        index = np.exp(index) / 1e3
    elif str_convertion == "e":
        index = np.exp(index)
    elif str_convertion == "K$":
        index = index / 1e3
    elif str_convertion == "%":
        index = index * 100

    index = index.map(("{:" + str_format + "}").format)

    df.index = pd.Index(index)

    return df
