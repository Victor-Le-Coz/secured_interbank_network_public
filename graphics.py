# import librairies
import os

# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from cProfile import label
import cpnet
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

# import modules
import functions as fct
import parameters as par
import emp_metrics as em


class ClassGraphics:
    def __init__(
        self,
        Dynamics,
        plot_period,
    ):
        self.Dynamics = Dynamics
        self.plot_period = plot_period

    def plot_all_trajectories(self):

        df_network_trajectory = self.Dynamics.df_network_trajectory
        df_bank_trajectory = self.Dynamics.df_bank_trajectory
        path_results = self.Dynamics.path_results

        # Plot aggregates
        items = [
            "loans",
            "central bank funding",
            "total assets",
            "deposits",
            "excess liquidity",
        ]
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=[f"{item} tot. network" for item in items],
            title="Macro-economic aggregates",
            file_name=f"{path_results}Macro-economic_aggregates.pdf",
        )

        # Plot collateral
        items = (
            [
                "securities usable",
                "securities encumbered",
            ]
            + par.off_bs_items
            + ["repo exposures"]
        )
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=[f"{item} tot. network" for item in items],
            title="Collateral aggregates",
            file_name=f"{path_results}collateral_aggregates.pdf",
        )

        # Plot the rate of collateral reuse in the network
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=["collateral reuse"],
            title="Collateral reuse",
            ylabel="number of reuse",
            file_name=f"{path_results}collateral_reuse.pdf",
        )

        # Plot gini
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=["gini"],
            title="Gini coefficient",
            ylabel="percentage",
            file_name=f"{path_results}gini.pdf",
        )

        # Plot reverse repo exposures statistics
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=[
                "repo exposures min network",
                "repo exposures max network",
                "repo exposures av. network",
            ],
            title="Repo exposures statistics",
            file_name=f"{path_results}repo_exposure_stats.pdf",
        )

        # Plot the average maturity of reverse repos transactions
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=["repos maturity av. network"],
            title="Repos maturity av. network",
            file_name=f"{path_results}average_maturity_repo_transactions.pdf",
        )

        # Plot average size of reverse repos transactions
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=["amount_ending_starting av. network"],
            title="Amount ending starting av. network",
            file_name=f"{path_results}average_size_repo_transactions.pdf",
        )

        # # plot average number of reverse repos transactions
        # self.plot_df_trajectory(
        #     df=df_network_trajectory,
        #     cols=["nb_ending_starting av. network"],
        #     title="Nb of repo transactions starting av. network",
        #     file_name=f"{path_results}average_nb_repo_transactions.pdf",
        # )

        # Plot the time series of the jaccard index
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=[
                f"jaccard index-{agg_period}" for agg_period in par.agg_periods
            ],
            title="Jaccard index",
            ylabel="percentage",
            file_name=f"{path_results}jaccard_index.pdf",
        )

        # Plot the time series of the network density
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=[
                f"network density-{agg_period}"
                for agg_period in par.agg_periods
            ],
            title="Network density",
            ylabel="percentage",
            file_name=f"{path_results}network_density.pdf",
        )

        # Plot the time series of the network average degree
        self.plot_trajectory(
            df=df_network_trajectory,
            cols=[
                f"av. degree-{agg_period}" for agg_period in par.agg_periods
            ],
            title="Av. degree",
            ylabel="degree",
            file_name=f"{path_results}average_degree.pdf",
        )

        # Plot the single bank trajectory time series.
        self.plot_df_bank_trajectory(df_bank_trajectory, path_results)

        days = range(
            self.Dynamics.Network.step + 1
        )  # to cover all steps up to now

        # plot the reverse repo network
        self.plot_weighted_adj_network(
            self.Dynamics.arr_reverse_repo_adj,
            self.Dynamics.arr_total_assets,
            days,
            f"{path_results}weighted_adj_network/",
            "reverse repo",
        )

        # plot degree distribution
        self.plot_degree_distribution(
            self.Dynamics.dic_in_degree,
            self.Dynamics.dic_out_degree,
            days,
            f"{path_results}degree_distribution/",
        )

        # Plot degree per asset
        self.plot_degree_per_asset(
            self.Dynamics.arr_total_assets,
            self.Dynamics.dic_degree,
            range(self.Dynamics.Network.nb_banks),
            days,
            f"{path_results}degree_per_asset/",
        )

        # Plot the core-periphery detection and assessment
        if self.Dynamics.cp_option:
            self.plot_cp_test(
                dic=self.Dynamics.dic_arr_binary_adj,
                algos=par.cp_algos,
                days=days,
                path_results=path_results,
                opt_agg=True,
            )
            self.plot_cp_test(
                dic=self.Dynamics.arr_reverse_repo_adj,
                algos=par.cp_algos,
                days=days,
                path_results=path_results,
                opt_agg=False,
            )
            self.Dynamics.p_value = 0  # to be modified

    def plot_trajectory(
        self,
        df,
        cols,
        title,
        file_name,
        ylabel="monetary units",
        figsize=par.small_figsize,
    ):
        fig = plt.figure(figsize=figsize)
        colors = sns.color_palette("flare", n_colors=len(cols))
        for i, col in enumerate(cols):
            plt.plot(
                df.index,
                df[col],
                color=colors[i],
            )
        plt.legend(cols)
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.savefig(f"{file_name}", bbox_inches="tight")
        plt.close()

    def plot_step_weighted_adj_network(
        self,
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
        log_node_sizes = [
            0 if i <= 1 else np.log(i) + 1 for i in ar_total_assets
        ]

        # define the position of the nodes
        pos = nx.spring_layout(bank_network)

        # draw the network
        plt.figure(figsize=figsize)
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
        self,
        arr_adj,
        arr_total_assets,
        days,
        path,
        name,
        figsize=par.small_figsize,
    ):

        fct.init_path(path)

        plot_steps = fct.get_plot_steps_from_period(days, self.plot_period)

        for step in plot_steps:
            self.plot_step_weighted_adj_network(
                arr_adj[step],
                arr_total_assets[step],
                days,
                step,
                path,
                name=name,
                figsize=figsize,
            )

    def plot_step_degree_distribution(
        self,
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
            [
                str(agg_period) + " time steps"
                for agg_period in par.agg_periods
            ],
            loc="upper left",
        )

        ax2.set_xlabel("degree")
        ax2.set_title("distribution of out-degree")
        ax2.legend(
            [
                str(agg_period) + " time steps"
                for agg_period in par.agg_periods
            ],
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
        self,
        dic_in_degree,
        dic_out_degree,
        days,
        path,
        figsize=par.small_figsize,
    ):

        fct.init_path(path)

        plot_steps = fct.get_plot_steps_from_period(days, self.plot_period)

        for step in plot_steps:
            self.plot_step_degree_distribution(
                dic_in_degree,
                dic_out_degree,
                days,
                step,
                path,
                figsize=figsize,
            )

    def plot_step_degree_per_asset(
        self,
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
        self,
        arr_total_assets,
        dic_degree,
        bank_ids,
        days,
        path,
        figsize=(6, 3),
        plot_days=False,
        finrep_days=False,
        finrep_bank_ids=False,
    ):

        fct.init_path(path)

        if plot_days:
            plot_steps = fct.get_plot_steps_from_days(days, plot_days)
            finrep_plot_steps = fct.get_plot_steps_from_days(
                finrep_days, plot_days
            )

        else:
            plot_steps = fct.get_plot_steps_from_period(days, self.plot_period)
            finrep_plot_steps = plot_steps

        for step, finrep_step in zip(plot_steps, finrep_plot_steps):
            self.plot_step_degree_per_asset(
                arr_total_assets[finrep_step],
                dic_degree,
                bank_ids,
                days,
                step,
                path,
                figsize=figsize,
                finrep_bank_ids=finrep_bank_ids,
            )

    def plot_step_cp_test(
        self,
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

    def run_step_cp_test(
        self,
        arr_reverse_repo_adj,
        algo,
        days,
        path_results,
        figsize=(6, 3),
    ):

        print(f"core-periphery tests using the {algo} approach")

        # initialise results and path
        sr_pvalue = pd.Series(dtype=float)
        fct.delete_n_init_path(path_results)

        plot_steps = fct.get_plot_steps_from_period(days, self.plot_period)

        for step in plot_steps:

            day = days[step]

            print(f"test on day {day}")

            # build nx object
            bank_network = nx.from_numpy_array(
                arr_reverse_repo_adj[step],
                parallel_edges=False,
                create_using=nx.DiGraph,
            )

            # run cpnet test
            sig_c, sig_x, significant, p_values = em.cpnet_test(
                bank_network, algo=algo
            )

            # store the p_value (only the first one)
            sr_pvalue.loc[day] = p_values[0]

            # plot
            if isinstance(day, pd.Timestamp):
                day_print = day.strftime("%Y-%m-%d")
            else:
                day_print = day

            self.plot_step_cp_test(
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

    def plot_cp_test(
        self,
        dic,
        algos,
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
                df_pvalue[algo] = self.run_step_cp_test(
                    dic_adj,
                    algo=algo,
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

    def plot_df_bank_trajectory(
        self, df_bank_trajectory, path, figsize=par.slide_figsize
    ):

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
            # "reverse repo exposures",
            # "own funds",
            # "repo exposures",
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
                df_bank_trajectory[f"av. in-degree-{agg_period}"],
                label=f"in-degree-{agg_period}",
                color=in_colors[i],
            )
        for i, agg_period in enumerate(par.agg_periods):
            ax3.plot(
                df_bank_trajectory.index,
                df_bank_trajectory[f"av. out-degree-{agg_period}"],
                label=f"out-degree-{agg_period}",
                color=out_colors[i],
            )

        keys = [
            "nb_ending_starting",
            "repos maturity av. bank",
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

    def plot_final_step(
        self,
    ):

        # Plot the break-down of the balance per bank
        self.plot_step_balance_sheet(
            self.Dynamics.Network.df_banks,
            self.Dynamics.path_results,
            self.Dynamics.Network.step,
        )

    def plot_step_balance_sheet(
        self,
        df_banks,
        path,
        step,
    ):
        plt.figure()
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=par.halfslide_figsize)

        ix_sorted = np.argsort(np.asarray(df_banks["total assets"]))
        banks_sorted = ["bank {}".format(str(b)) for b in ix_sorted]

        asset_bottoms = [np.zeros(len(banks_sorted))]
        for col in par.assets[:-1]:
            asset_bottoms.append(
                asset_bottoms[-1] + df_banks.loc[ix_sorted, col]
            )
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
        ax1.set_title(f"Assets on day {step}")

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
        ax2.set_title(f"Liabilities on day {step}")

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
        ax3.set_title(f"Off-balance sheet items on day {step}")

        # plt.subplots_adjust(hspace=0.3)
        fig.tight_layout()
        plt.savefig(
            f"{path}balance_sheets_on_day_{step}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_multiple_key(
    param_values,
    input_param,
    output,
    path,
    short_key,
    nb_char,
    figsize=par.small_figsize,
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
        [str(agg_period) + " time steps" for agg_period in par.agg_periods],
        loc="upper left",
    )
    plt.title(short_key + "x periods as a fct. of " + input_param)
    fig.tight_layout()
    plt.savefig(
        path + short_key + "_agg_" + input_param + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_single_key(
    key, param_values, input_param, output, path, figsize=par.small_figsize
):
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
    param_values, input_param, output, jaccard_periods, path
):
    output = fct.reformat_output(output)

    # plot multiple keys on the same chart
    for short_key in par.output_mlt_keys:
        plot_multiple_key(
            param_values=param_values,
            input_param=input_param,
            output=output,
            path=path,
            short_key=short_key,
            nb_char=len(short_key),
        )

    # case of the jaccard old version (with jaccard perriods != agg periodes)
    plot_multiple_key(
        param_values=param_values,
        input_param=input_param,
        output=output,
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
