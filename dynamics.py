import os

os.environ["OMP_NUM_THREADS"] = "1"

import networkx as nx
import numpy as np
from scipy.stats import pareto
from tqdm import tqdm
import graphics as gx
import shocks as sh
import functions as fct
from network import ClassNetwork
import pandas as pd
import parameters as par


class ClassDynamics:
    def __init__(
        self,
        Network,
        nb_steps,
        path_results,
        agg_periods,
        jaccard_periods,
        cp_option=False,
    ):
        # initialization of the class parameters.
        self.Network = Network
        self.nb_steps = nb_steps
        self.path_results = path_results
        self.agg_periods = agg_periods
        self.cp_option = cp_option

        # Create the required path to store the results
        fct.init_results_path(self.path_results)

        # Definition of the internal parameters of ClassDynamics
        self.banks_degree = np.zeros(
            self.Network.nb_banks
        )  # Numpy array of the degree of the banks in the network.

        # individual trajectory
        self.df_bank_trajectory = pd.DataFrame(
            index=range(self.nb_steps), columns=par.bank_metrics
        )
        self.single_bank_id = 0  # the selected single bank id

        # total network trajectories
        columns = par.network_metrics + [
            f"{link_network_metric}-{agg_period}"
            for link_network_metric in par.link_network_metrics
            for agg_period in self.agg_periods
        ]
        self.df_network_trajectory = pd.DataFrame(
            0, index=range(self.nb_steps), columns=columns
        )

        # definition of the p-value parameter of the core-perihpery structure dected by the cpnet algo
        self.p_value = 1  # initialization at 1 (non significant test)

        self.agg_binary_adj_dic = (
            {}
        )  # dictionary of the aggregated ajency matrix over a given period
        for agg_period in self.agg_periods:
            self.agg_binary_adj_dic.update(
                {
                    agg_period: np.zeros(
                        (self.Network.nb_banks, self.Network.nb_banks)
                    )
                }
            )

        self.prev_agg_binary_adj_dic = (
            {}
        )  # dictionary of the previous aggregated binary adjency matrices for different aggregation periods
        for agg_period in self.agg_periods:
            self.prev_agg_binary_adj_dic.update(
                {
                    agg_period: np.zeros(
                        (self.Network.nb_banks, self.Network.nb_banks)
                    )
                }
            )

        self.prev_binary_adj_dic = (
            {}
        )  # dictionary of the previous adjacency matrix, used for the computation of the jaccard index (stable trading relationships) of different time length
        for agg_period in agg_periods:
            self.prev_binary_adj_dic.update(
                {
                    agg_period: np.zeros(
                        (self.Network.nb_banks, self.Network.nb_banks)
                    )
                }
            )

    def record_trajectories(self):
        """
        Instance method allowing the computation of the time_series_metrics
        parameter as well as the network_assets, network_liabilities and
        network_off_balance dictionaries.
        :return:
        """

        # Build the time series of the accounting items
        for item in par.bank_items:
            self.df_network_trajectory.loc[
                self.Network.step, f"{item} tot. volume"
            ] = self.Network.df_banks[item].sum()

        # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
        if self.Network.step > 0:
            for agg_period in self.agg_periods:
                if self.Network.step % agg_period > 0:
                    self.agg_binary_adj_dic.update(
                        {
                            agg_period: np.logical_or(
                                self.Network.dic_matrices["binary_adjency"],
                                self.agg_binary_adj_dic[agg_period],
                            )
                        }
                    )
                elif self.Network.step % agg_period == 0:
                    self.agg_binary_adj_dic.update(
                        {
                            agg_period: self.Network.dic_matrices[
                                "binary_adjency"
                            ]
                        }
                    )

        # Build the time series of the Av. nb. of repo transactions ended (2/2).
        self.df_network_trajectory.loc[
            self.Network.step, "Av. nb. of repo transactions ended"
        ] = self.Network.df_banks["nb_ending_starting"].mean()

        # Build the time series of the Average volume of repo transaction ended within a step (2/2).
        nb_trans = self.Network.df_banks["nb_ending_starting"].sum()
        if nb_trans > 0:
            self.df_network_trajectory.loc[
                self.Network.step,
                "Av. volume of repo transactions ended",
            ] = (
                self.Network.df_banks["amount_ending_starting"].sum()
                / nb_trans
            )
        else:
            self.df_network_trajectory.loc[
                self.Network.step,
                "Av. volume of repo transactions ended",
            ] = 0

        # Build the time series of the weighted average maturity of the repo transactions (2/2)
        ending_amount = self.Network.df_banks["ending_amount"].sum()
        if ending_amount > 0:
            self.df_network_trajectory.loc[
                self.Network.step, "Repos av. maturity"
            ] = (
                self.Network.df_banks["maturity@ending_amount"].sum()
                / ending_amount
            )
        else:
            self.df_network_trajectory.loc[
                self.Network.step, "Repos av. maturity"
            ] = 0

        # Build the average in-degree in the network.
        bank_network = nx.from_numpy_matrix(
            self.Network.dic_matrices["binary_adjency"],
            parallel_edges=False,
            create_using=nx.DiGraph,
        )  # first define a networkx object.
        self.df_network_trajectory.loc[
            self.Network.step, "Av. in-degree"
        ] = np.array(bank_network.in_degree())[:, 1].mean()

        # Build the jaccard index time series - version non aggregated.
        for agg_period in self.agg_periods:
            if self.Network.step > 0 and self.Network.step % agg_period == 0:

                self.df_network_trajectory.loc[
                    self.Network.step, f"Jaccard index-{agg_period}"
                ] = (
                    np.logical_and(
                        self.Network.dic_matrices["binary_adjency"],
                        self.prev_binary_adj_dic[agg_period],
                    ).sum()
                    / np.logical_or(
                        self.Network.dic_matrices["binary_adjency"],
                        self.prev_binary_adj_dic[agg_period],
                    ).sum()
                )
                self.prev_binary_adj_dic.update(
                    {
                        agg_period: self.Network.dic_matrices[
                            "binary_adjency"
                        ].copy()
                    }
                )
            elif self.Network.step > 0:
                self.df_network_trajectory.loc[
                    self.Network.step, f"Jaccard index-{agg_period}"
                ] = self.df_network_trajectory.loc[
                    self.Network.step - 1, f"Jaccard index-{agg_period}"
                ]

        # Build the jaccard index time series - version aggregated.
        for agg_period in self.agg_periods:
            if self.Network.step % agg_period == agg_period - 1:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"Jaccard index-{agg_period}",
                ] = (
                    np.logical_and(
                        self.agg_binary_adj_dic[agg_period],
                        self.prev_agg_binary_adj_dic[agg_period],
                    ).sum()
                    / np.logical_or(
                        self.agg_binary_adj_dic[agg_period],
                        self.prev_agg_binary_adj_dic[agg_period],
                    ).sum()
                )
                self.prev_agg_binary_adj_dic.update(
                    {agg_period: self.agg_binary_adj_dic[agg_period].copy()}
                )
            elif self.Network.step > 0:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"Jaccard index-{agg_period}",
                ] = self.df_network_trajectory.loc[
                    self.Network.step - 1,
                    f"Jaccard index-{agg_period}",
                ]

        # Build the network density indicator.
        for agg_period in self.agg_periods:
            if self.Network.step % agg_period == agg_period - 1:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"Network density-{agg_period}",
                ] = self.agg_binary_adj_dic[agg_period].sum() / (
                    self.Network.nb_banks * (self.Network.nb_banks - 1.0)
                )  # for a directed graph
            elif self.Network.step > 0:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"Network density-{agg_period}",
                ] = self.df_network_trajectory.loc[
                    self.Network.step - 1,
                    f"Network density-{agg_period}",
                ]

        # Build the gini coeficient of the network
        self.df_network_trajectory.loc[self.Network.step, "Gini"] = fct.gini(
            self.Network.df_banks["Total assets"]
        )

        # Build the statistics regarding the size of the reverse repos across the network at a given time step
        if len(self.Network.dic_matrices["non-zero_adjency"]) == 0:
            self.df_network_trajectory.loc[
                self.Network.step, "Repos exposure min volume"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "Repos exposure max volume"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "Repos exposure av. volume"
            ] = 0
        else:
            self.df_network_trajectory.loc[
                self.Network.step, "Repos exposure min volume"
            ] = np.min(self.Network.dic_matrices["non-zero_adjency"])
            self.df_network_trajectory.loc[
                self.Network.step, "Repos exposure max volume"
            ] = np.max(self.Network.dic_matrices["non-zero_adjency"])
            self.df_network_trajectory.loc[
                self.Network.step, "Repos exposure av. volume"
            ] = np.mean(self.Network.dic_matrices["non-zero_adjency"])

        # build the time serrie of Collateral reuse
        self.df_network_trajectory.loc[
            self.Network.step, "Collateral reuse"
        ] = (
            self.df_network_trajectory.loc[
                self.Network.step, "Securities Reused tot. volume"
            ]
        ) / (
            self.df_network_trajectory.loc[
                self.Network.step, "Securities Collateral tot. volume"
            ]
            + 1e-10
        )

        # Build the dictionary of the degree (total of in and out) of each node in the network at a given step
        self.banks_degree = np.array(bank_network.degree())[:, 1]

    def fill_df_bank_trajectory(self):

        # defin the single bank that we want to plot
        Bank = self.Network.banks[self.single_bank_id]

        # Build the time series of the accounting items
        for item in par.bank_items:
            self.df_bank_trajectory.loc[
                self.Network.step, item
            ] = self.Network.df_banks.loc[self.single_bank_id, item]

        # In and Out-degree
        bank_network = nx.from_numpy_matrix(
            self.Network.dic_matrices["binary_adjency"],
            parallel_edges=False,
            create_using=nx.DiGraph,
        )
        self.df_bank_trajectory.loc[
            self.Network.step, "Av. in-degree"
        ] = bank_network.in_degree(self.single_bank_id)
        self.df_bank_trajectory.loc[
            self.Network.step, "Av. out-degree"
        ] = bank_network.out_degree(self.single_bank_id)

        # size of transactions of end repos per step
        nb_trans = self.Network.df_banks.loc[
            self.single_bank_id, "nb_ending_starting"
        ]
        if nb_trans > 0:
            self.df_bank_trajectory.loc[
                self.Network.step,
                "Av. volume of repo transactions ended",
            ] = (
                self.Network.df_banks.loc[
                    self.single_bank_id, "amount_ending_starting"
                ]
                / nb_trans
            )
        else:
            self.df_bank_trajectory.loc[
                self.Network.step,
                "Av. volume of repo transactions ended",
            ] = 0

        # weighted average maturity of repos at a given step for a given bank
        ending_amount = self.Network.df_banks.loc[
            self.single_bank_id, "ending_amount"
        ]
        if ending_amount > 0:
            self.df_bank_trajectory.loc[
                self.Network.step, "Repos av. maturity"
            ] = (
                self.Network.df_banks.loc[
                    self.single_bank_id, "maturity@ending_amount"
                ]
                / ending_amount
            )
        else:
            self.df_bank_trajectory.loc[
                self.Network.step, "Repos av. maturity"
            ] = 0

    def store_trajectories(self):
        """
        Instance method saving all the figures representing the network
        status at a given time-step as well as all the time series plots of the chosen metrics
        :return:
        """

        # save the data frame results
        self.df_bank_trajectory.to_csv(
            f"{self.path_results}df_bank_trajectory.csv"
        )
        self.df_network_trajectory.to_csv(
            f"{self.path_results}df_network_trajectory.csv"
        )

        gx.plot_network(
            adj=self.Network.dic_matrices["adjency"],
            network_total_assets=self.Network.df_banks["Total assets"],
            path=self.path_results + "repo_networks/",
            step=self.Network.step,
            name_in_title="reverse repo",
        )
        fct.save_np_array(
            self.Network.dic_matrices["adjency"],
            self.path_results + "repo_networks/adj_matrix",
        )

        # Plot the trust network
        gx.plot_network(
            adj=self.Network.dic_matrices["trust"].T
            / (self.Network.dic_matrices["trust"].std() + 1e-8),
            network_total_assets=self.Network.df_banks["Total assets"],
            path=self.path_results + "trust_networks/",
            step=self.Network.step,
            name_in_title="trust",
        )
        fct.save_np_array(
            self.Network.dic_matrices["trust"],
            self.path_results + "trust_networks/trust",
        )

        # Plot the break-down of the balance per bank
        gx.bar_plot_balance_sheet(
            self.Network.df_banks,
            self.path_results + "balance_Sheets/",
            self.Network.step,
        )

        # Plot the break-down of the deposits per bank in relative shares
        gx.bar_plot_deposits(
            self.Network.df_banks["Deposits"],
            self.path_results + "deposits/",
            self.Network.step,
        )

        # Plot the core-periphery detection and assessment
        # special case here, an intermediary computation to keep track of p-values
        if self.cp_option:
            if self.Network.step > 0:
                bank_network = nx.from_numpy_matrix(
                    self.Network.dic_matrices["binary_adjency"],
                    parallel_edges=False,
                    create_using=nx.DiGraph,
                )  # build nx object
                sig_c, sig_x, significant, p_value = fct.cpnet_test(
                    bank_network
                )  # run cpnet test
                self.p_value = p_value  # record p_value
                gx.plot_core_periphery(
                    bank_network=bank_network,
                    sig_c=sig_c,
                    sig_x=sig_x,
                    path=self.path_results + "core-periphery_structure/",
                    step=self.Network.step,
                    name_in_title="reverse repos",
                )  # plot charts

        # Plot the link between centrality and total asset size
        gx.plot_asset_per_degree(
            self.Network.df_banks["Total assets"],
            self.banks_degree,
            self.path_results,
        )

        # Plot the time series of the total repos in the network
        gx.plot_repos(
            self.df_network_trajectory,
            self.path_results,
        )

        # Plot the time series of the total MROS and loans in the network
        gx.plot_assets_loans_mros(
            self.df_network_trajectory,
            self.path_results,
        )

        # Plot the time series of the securities usable, encumbered and
        # re-used in the network
        gx.plot_collateral(self.df_network_trajectory, self.path_results)

        # Plot the time series of the weighted average number of time the
        # collateral is reused in the network
        gx.plot_collateral_reuse(
            self.df_network_trajectory,
            self.path_results,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_not_aggregated(
            self.df_network_trajectory,
            self.path_results,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_aggregated(
            self.df_network_trajectory,
            self.path_results,
        )

        # Plot the time series of the total excess liquidity and deposits in
        # the network
        gx.plot_excess_liquidity_and_deposits(
            self.df_network_trajectory, self.path_results
        )

        # Plot the time series of the network density
        gx.plot_network_density(
            self.df_network_trajectory,
            self.path_results,
        )

        # Plot the time series of the gini coefficients
        gx.plot_gini(self.df_network_trajectory, self.path_results)

        # Plot the time series of the statistics of the size of reverse repo
        gx.plot_reverse_repo_size_stats(
            self.df_network_trajectory, self.path_results
        )

        # Plot the time series of the network average degree
        gx.plot_degre_network(self.df_network_trajectory, self.path_results)

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_nb_transactions(
            self.df_network_trajectory, self.path_results
        )

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_size_transactions(
            self.df_network_trajectory, self.path_results
        )

        # Plot the average maturity of repos.
        gx.plot_average_maturity_repo(
            self.df_network_trajectory, self.path_results
        )

        # Plot the single bank trajectory time series.
        gx.plot_df_bank_trajectory(self.df_bank_trajectory, self.path_results)

    def simulate(self, save_every, output_keys=False):

        # record and store trajectories & parameters used at step 0
        self.save_param(save_every)
        self.record_trajectories()
        self.fill_df_bank_trajectory()
        self.store_trajectories()

        # simulate the network
        for _ in tqdm(range(self.nb_steps - 1)):

            # update one step
            self.Network.step_network()

            # record trajectories
            self.record_trajectories()
            self.fill_df_bank_trajectory()

            # store only every given steps
            if self.Network.step % save_every == 0:
                self.store_trajectories()

        # store the final step (if not already done)
        if self.nb_steps % save_every != 0:
            self.store_trajectories()

        # final print
        self.print_summary()

        if output_keys:
            output = self.build_output(output_keys)
            return output

    def save_param(self, save_every):
        with open(self.path_results + "param.txt", "w") as f:
            f.write(
                (
                    "nb_banks={} \n"
                    "alpha={} \n"
                    "beta_init={} \n"
                    "beta_reg={} \n"
                    "beta_star={} \n"
                    "gamma={} \n"
                    "collateral_value={} \n"
                    "initialization_method={} \n"
                    "alpha_pareto={} \n"
                    "shock_method={} \n"
                    "shocks_law={} \n"
                    "shocks_vol={} \n"
                    "result_location={} \n"
                    "min_repo_size={} \n"
                    "nb_steps={} \n"
                    "save_every={} \n"
                    "jaccard_periods={} \n"
                    "LCR_mgt_opt={} \n"
                ).format(
                    self.Network.nb_banks,
                    self.Network.alpha,
                    self.Network.beta_init,
                    self.Network.beta_reg,
                    self.Network.beta_star,
                    self.Network.gamma,
                    self.Network.collateral_value,
                    self.Network.initialization_method,
                    self.Network.alpha_pareto,
                    self.Network.shocks_method,
                    self.Network.shocks_law,
                    self.Network.shocks_vol,
                    self.path_results,
                    self.Network.min_repo_size,
                    self.nb_steps,
                    save_every,
                    self.agg_periods,
                    self.Network.LCR_mgt_opt,
                )
            )

    def build_output(self):
        output = {}
        stat_len_step = 250

        # build the time series metrics outputs
        for metric in self.df_network_trajectory.columns:
            output.update(
                {
                    metric: self.df_network_trajectory.loc[
                        -stat_len_step:, metric
                    ].mean()
                }
            )

        output.update({"Core-Peri. p_val.": self.p_value})

        return output

    def print_summary(self):
        # new approach
        print(
            "repos av. maturity: {}".format(
                {
                    self.df_network_trajectory.loc[
                        self.Network.step, "Repos av. maturity"
                    ]
                }
            )
        )
        print(
            "Repos exposure av. volume: {}".format(
                {
                    self.df_network_trajectory.loc[
                        self.Network.step, "Repos exposure av. volume"
                    ]
                }
            )
        )
