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


class ClassGraphics:
    def __init__(
        self,
        Dynamics,
        plot_period,
    ):
        self.Dynamics = Dynamics
        self.plot_period = plot_period

    def plot_all_trajectories(self):

        # create paths
        path_results = self.Dynamics.path_results
        os.makedirs(f"{path_results}accounting_view/", exist_ok=True)
        os.makedirs(f"{path_results}transaction_view/", exist_ok=True)
        os.makedirs(f"{path_results}exposure_view/", exist_ok=True)

        # --------------
        # all views

        # plot all trajectories
        for index in par.df_figures.index:
            self.plot_trajectory(
                df=self.Dynamics.df_network_trajectory,
                cols=[
                    f"{item}{par.df_figures.loc[index,'extension']}"
                    for item in par.df_figures.loc[index, "items"]
                ],
                file_name=f"{path_results}{index}.pdf",
            )

        # --------------
        # exposure view - specific cases

        # Plot the single bank trajectory time series.
        self.plot_df_bank_trajectory(
            self.Dynamics.df_bank_trajectory,
            f"{path_results}exposure_view/",
        )

        days = range(
            self.Dynamics.Network.step + 1
        )  # to cover all steps up to now

        # plot the reverse repo network
        self.plot_weighted_adj_network(
            self.Dynamics.arr_rev_repo_exp_adj,
            self.Dynamics.arr_total_assets,
            days,
            f"{path_results}exposure_view/weighted_adj_network/",
            "reverse repo",
        )

        # plot degree distribution
        self.plot_degree_distribution(
            self.Dynamics.dic_in_degree,
            self.Dynamics.dic_out_degree,
            days,
            f"{path_results}exposure_view/degree_distribution/",
        )

        # Plot degree per asset
        self.plot_degree_per_asset(
            self.Dynamics.arr_total_assets,
            self.Dynamics.dic_degree,
            range(self.Dynamics.Network.nb_banks),
            days,
            f"{path_results}exposure_view/degree_per_asset/",
        )

        # Plot the core-periphery detection and assessment
        if self.Dynamics.cp_option:
            self.plot_cp_test(
                dic=self.Dynamics.dic_arr_binary_adj,
                algos=par.cp_algos,
                days=days,
                path_results=f"{path_results}exposure_view/",
                opt_agg=True,
            )
            self.plot_cp_test(
                dic=self.Dynamics.arr_rev_repo_exp_adj,
                algos=par.cp_algos,
                days=days,
                path_results=f"{path_results}exposure_view/",
                opt_agg=False,
            )
            self.Dynamics.p_value = 0  # to be modified

    def plot_trajectory(
        self,
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
        self,
        arr_adj,
        arr_total_assets,
        days,
        path,
        name,
        figsize=par.small_figsize,
    ):

        os.makedirs(path, exist_ok=True)

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

        os.makedirs(path, exist_ok=True)

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

        os.makedirs(path, exist_ok=True)

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
        figsize=(6, 3),
    ):

        os.makedirs(path, exist_ok=True)

        # Visualization
        fig, ax = plt.subplots(figsize=figsize)
        ax = plt.gca()
        ax, pos = cpnet.draw(bank_network, sig_c, sig_x, ax)

        # show the plot
        fig.tight_layout()
        plt.savefig(
            f"{path}step_core-periphery_structure_{step}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def run_algo_cp_test(
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

            if sig_c != {}:
                self.plot_step_cp_test(
                    bank_network=bank_network,
                    sig_c=sig_c,
                    sig_x=sig_x,
                    path=f"{path_results}",
                    step=day_print,
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
                df_pvalue[algo] = self.run_algo_cp_test(
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

    def plot_final_step(
        self,
    ):

        df_banks = self.Dynamics.Network.df_banks
        step = self.Dynamics.Network.step
        path = f"{self.Dynamics.path_results}accounting_view/"

        # Plot the break-down of the balance per bank
        self.plot_step_balance_sheet(
            df_banks,
            path,
            step,
        )

        # plot all power law tests
        self.plot_all_power_law_tests(df_banks, f"{path}power_laws/")

    def plot_step_balance_sheet(
        self,
        df_banks,
        path,
        step,
    ):

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

    def power_law_test(self, sr_data, file_name, figsize=par.small_figsize):

        # fit the data with the powerlaw librairy

        if (len(sr_data.dropna()) > 1) and (
            sr_data.abs().sum() > par.float_limit
        ):  # at least 2 data points required with non negligeable size
            fit = powerlaw.Fit(sr_data.dropna())

            # define the figure and colors
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            colors = sns.color_palette("flare", n_colors=3)
            col = sr_data.name

            # ax1 : pdf
            try:
                fit.plot_pdf(color=colors[0], ax=ax1)
            except:
                sr_data.to_csv("./support/sr_data.csv")
                sys.exit(3)

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

    def plot_all_power_law_tests(
        self, df_banks, path, figsize=par.small_figsize
    ):

        # replace all values smaller or equal to 0 by nan (to avoid powerlaw warnings)
        df = df_banks.mask(df_banks <= 0)

        # build path
        os.makedirs(f"{path}", exist_ok=True)

        df_power_law = pd.DataFrame(
            index=df.columns, columns=["alpha_power_law", "p_value"]
        )

        for col in df.columns:

            # define data
            sr_data = df[col]

            # check that some data exists
            if not (sr_data.isna().all()):

                # run and plot the power law test
                df_power_law.loc[col] = self.power_law_test(
                    sr_data, f"{path}{col}", figsize=figsize
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
        ax2.plot(
            df_power_law.index, df_power_law["p_value"], ".", color=colors[1]
        )
        ax2.set_ylabel("p-value")
        ax2.legend(["p-value"], loc="upper right")

        plt.grid()
        plt.savefig(f"{path}power_law_tests.pdf", bbox_inches="tight")
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
    plt.legend(par.df_plt.loc[cols, "legend"])
    plt.xlabel(par.df_plt.loc[input_parameter, "label"])
    plt.ylabel(par.df_plt.loc[cols[0], "label"])
    plt.xscale(par.df_plt.loc[input_parameter, "scale"])
    plt.yscale(par.df_plt.loc[cols[0], "scale"])
    plt.grid()
    plt.savefig(f"{file_name}", bbox_inches="tight")
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
