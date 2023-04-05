import os

os.environ["OMP_NUM_THREADS"] = "1"

import networkx as nx
import numpy as np
from scipy.stats import pareto
from tqdm import tqdm
import graphics as gx
from bank import ClassBank
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

        # Definition of the dictionaries associating to each of the accounting
        # items, its corresponding numpy array of its value per bank,
        # at a given time step.
        self.dic_banks_liabilities = {
            "Own Funds": np.zeros(self.Network.nb_banks),
            "Deposits": np.zeros(self.Network.nb_banks),
            "Repos": np.zeros(self.Network.nb_banks),
            "MROs": np.zeros(self.Network.nb_banks),
        }
        self.dic_banks_assets = {
            "Cash": np.zeros(self.Network.nb_banks),
            "Securities Usable": np.zeros(self.Network.nb_banks),
            "Securities Encumbered": np.zeros(self.Network.nb_banks),
            "Loans": np.zeros(self.Network.nb_banks),
            "Reverse Repos": np.zeros(self.Network.nb_banks),
        }
        self.dic_banks_off_balance = {
            "Securities Collateral": np.zeros(self.Network.nb_banks),
            "Securities Reused": np.zeros(self.Network.nb_banks),
        }

        # Definition of the adjacency matrices
        self.adj_matrix = np.zeros(
            (self.Network.nb_banks, self.Network.nb_banks)
        )  # reverse repos
        # adjacency matrix
        self.trust_adj_matrix = np.zeros(
            (self.Network.nb_banks, self.Network.nb_banks)
        )  # trust
        # coeficients adjacency matrix

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

    def fill_df_network_trajectory(self):

        # initialization of the list used to compute the weighted average repo maturity
        weighted_repo_maturity = []
        total_repo_amount = 0

        # initialization of the counter of the repo transactions ended within a step across all banks
        total_repo_transactions_counter = 0

        # initialization of the total amount of the repo transactions ended ended within a step across all banks
        total_repo_transactions_size = 0

        # Loop over the banks and over the accounting items time series
        weighted_repo_maturity = []
        for i, Bank in enumerate(self.Network.Banks):

            # Build the time series of the accounting items and store the
            # network dictionaries of the accounting items values
            for key in Bank.assets.keys():  # only loop over assets items.
                self.df_network_trajectory.loc[
                    self.Network.step, key + " tot. volume"
                ] += Bank.assets[key]
                # Computes the total of a given item at a given time step.
                self.dic_banks_assets[key][i] = Bank.assets[key]  # Fill-in
                # the value of each accounting item of each bank into the
                # network asset dictionary.
            for key in Bank.liabilities.keys():  # only loop over liabilities
                # items.
                self.df_network_trajectory.loc[
                    self.Network.step, key + " tot. volume"
                ] += Bank.liabilities[key]
                self.dic_banks_liabilities[key][i] = Bank.liabilities[key]
            for key in Bank.off_balance.keys():  # only loop over off-balance
                # items.
                self.df_network_trajectory.loc[
                    self.Network.step, key + " tot. volume"
                ] += Bank.off_balance[key]
                self.dic_banks_off_balance[key][i] = Bank.off_balance[key]

            # Build the adjacency matrix of the reverse repos
            self.adj_matrix[i, :] = np.array(
                list(self.Network.Banks[i].reverse_repos.values())
            )

            # Build the adjacency matrix of the trust coefficients
            trusts = np.array(list(self.Network.Banks[i].trust.values()))
            self.trust_adj_matrix[i, :i] = trusts[:i]
            self.trust_adj_matrix[i, i + 1 :] = trusts[i:]

            # Build the total assets of each bank
            self.Network.banks_total_assets[i] = self.Network.Banks[
                i
            ].total_assets()

            # Build the deposits numpy array of each bank
            self.Network.banks_deposits[i] = self.Network.Banks[i].liabilities[
                "Deposits"
            ]

            # Build the total network excess liquidity time series
            self.df_network_trajectory.loc[
                self.Network.step, "Excess Liquidity"
            ] += (
                self.Network.Banks[i].assets["Cash"]
                - self.Network.Banks[i].alpha
                * self.Network.Banks[i].liabilities["Deposits"]
            )

            # Build the weighted average maturity of repos (1/2).
            weighted_repo_maturity += list(
                np.array(self.Network.Banks[i].repos_on_maturities)
                * np.array(self.Network.Banks[i].repos_on_amounts)
            )  # add the on balance repos
            weighted_repo_maturity += list(
                np.array(self.Network.Banks[i].repos_off_maturities)
                * np.array(self.Network.Banks[i].repos_off_amounts)
            )
            total_repo_amount += sum(
                self.Network.Banks[i].repos_on_amounts
            ) + sum(
                self.Network.Banks[i].repos_off_amounts
            )  # add the off balance repos

            # Build the time series of the Average number of repo transaction ended within a step (1/2).
            total_repo_transactions_counter += self.Network.Banks[
                i
            ].repo_transactions_counter  # compute the sum

            # Build the time series of the Average size of repo transaction ended within a step (1/2).
            total_repo_transactions_size += self.Network.Banks[
                i
            ].repo_transactions_size  # compute the sum

            # Build the time serie of the total assets across all banks
            self.df_network_trajectory.loc[
                self.Network.step, "Assets tot. volume"
            ] += Bank.total_assets()

        return (
            weighted_repo_maturity,
            total_repo_amount,
            total_repo_transactions_counter,
            total_repo_transactions_size,
        )

    def record_trajectories(self):
        """
        Instance method allowing the computation of the time_series_metrics
        parameter as well as the network_assets, network_liabilities and
        network_off_balance dictionaries.
        :return:
        """

        # fill the df_network_metrics data frame for the current step
        (
            weighted_repo_maturity,
            total_repo_amount,
            total_repo_transactions_counter,
            total_repo_transactions_size,
        ) = self.fill_df_network_trajectory()

        # clean the adj matrix from the negative values (otherwise the algo generate -1e-14 values for the reverse repos)
        self.adj_matrix[self.adj_matrix < 0] = 0

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(
            self.adj_matrix > self.Network.min_repo_size, True, False
        )

        # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
        if self.Network.step > 0:
            for agg_period in self.agg_periods:
                if self.Network.step % agg_period > 0:
                    self.agg_binary_adj_dic.update(
                        {
                            agg_period: np.logical_or(
                                binary_adj,
                                self.agg_binary_adj_dic[agg_period],
                            )
                        }
                    )
                elif self.Network.step % agg_period == 0:
                    self.agg_binary_adj_dic.update({agg_period: binary_adj})

        # Build the time series of the Av. nb. of repo transactions ended (2/2).
        self.df_network_trajectory.loc[
            self.Network.step, "Av. nb. of repo transactions ended"
        ] = (total_repo_transactions_counter / self.Network.nb_banks)

        # Build the time series of the Average volume of repo transaction ended within a step (2/2).
        if total_repo_transactions_counter != 0:
            self.df_network_trajectory.loc[
                self.Network.step, "Av. volume of repo transactions ended"
            ] = (
                total_repo_transactions_size / total_repo_transactions_counter
            )
        else:
            self.df_network_trajectory.loc[
                self.Network.step, "Av. volume of repo transactions ended"
            ] = 0

        # Build the time series of the weighted average maturity of the repo transactions (2/2)
        self.df_network_trajectory.loc[
            self.Network.step, "Repos av. maturity"
        ] = (np.sum(weighted_repo_maturity) / total_repo_amount)

        # Build the average in-degree in the network.
        bank_network = nx.from_numpy_matrix(
            binary_adj,
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
                        binary_adj,
                        self.prev_binary_adj_dic[agg_period],
                    ).sum()
                    / np.logical_or(
                        binary_adj,
                        self.prev_binary_adj_dic[agg_period],
                    ).sum()
                )
                self.prev_binary_adj_dic.update(
                    {agg_period: binary_adj.copy()}
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
            self.Network.banks_total_assets
        )

        # Build the statistics regarding the size of the reverse repos across the network at a given time step
        non_zero_adj_matrix = self.adj_matrix[
            np.nonzero(self.adj_matrix)
        ]  # keep only non zero reverse repos

        if len(non_zero_adj_matrix) == 0:
            self.df_network_trajectory.loc[
                self.Network.step, "Repos min volume"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "Repos max volume"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "Repos av. volume"
            ] = 0
        else:
            self.df_network_trajectory.loc[
                self.Network.step, "Repos min volume"
            ] = np.min(non_zero_adj_matrix)
            self.df_network_trajectory.loc[
                self.Network.step, "Repos max volume"
            ] = np.max(non_zero_adj_matrix)
            self.df_network_trajectory.loc[
                self.Network.step, "Repos av. volume"
            ] = np.mean(non_zero_adj_matrix)

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
        Bank = self.Network.Banks[self.single_bank_id]

        # Build the time series of the accounting item of the bank bank_id
        for key in Bank.assets.keys():
            self.df_bank_trajectory.loc[self.Network.step, key] = Bank.assets[
                key
            ]
        for key in Bank.liabilities.keys():
            self.df_bank_trajectory.loc[
                self.Network.step, key
            ] = Bank.liabilities[key]
        for key in Bank.off_balance.keys():
            self.df_bank_trajectory.loc[
                self.Network.step, key
            ] = Bank.off_balance[key]

        # In and Out-degree
        binary_adj = np.where(
            self.adj_matrix > self.Network.min_repo_size, True, False
        )
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )
        self.df_bank_trajectory.loc[
            self.Network.step, "Av. in-degree"
        ] = bank_network.in_degree(self.single_bank_id)
        self.df_bank_trajectory.loc[
            self.Network.step, "Av. out-degree"
        ] = bank_network.out_degree(self.single_bank_id)

        # Number of transactions of end repos per step
        self.df_bank_trajectory.loc[
            self.Network.step, "Nb. of repo transactions ended"
        ] = self.Network.Banks[self.single_bank_id].repo_transactions_counter

        # size of transactions of end repos per step
        if (
            self.Network.Banks[self.single_bank_id].repo_transactions_counter
            != 0
        ):
            self.df_bank_trajectory.loc[
                self.Network.step, "Av. volume of repo transactions ended"
            ] = (
                self.Network.Banks[self.single_bank_id].repo_transactions_size
                / self.Network.Banks[
                    self.single_bank_id
                ].repo_transactions_counter
            )
        else:
            self.df_bank_trajectory.loc[
                self.Network.step, "Av. volume of repo transactions ended"
            ] = 0

        # weighted average maturity of repos at a given step for a given bank
        self.df_bank_trajectory.loc[
            self.Network.step, "Repos av. maturity"
        ] = np.sum(
            list(
                np.array(
                    self.Network.Banks[self.single_bank_id].repos_on_maturities
                )
                * np.array(
                    self.Network.Banks[self.single_bank_id].repos_on_amounts
                )
            )
            + list(
                np.array(
                    self.Network.Banks[
                        self.single_bank_id
                    ].repos_off_maturities
                )
                * np.array(
                    self.Network.Banks[self.single_bank_id].repos_off_amounts
                )
            )
        ) / (
            sum(self.Network.Banks[self.single_bank_id].repos_on_amounts)
            + sum(self.Network.Banks[self.single_bank_id].repos_off_amounts)
        )

    def final_print(self):

        # Print the weighted average maturity of repos
        weighted_repo_maturity = []
        total_repo_amount = 0
        for i, bank in enumerate(self.Network.Banks):
            weighted_repo_maturity += list(
                np.array(bank.repos_on_maturities)
                * np.array(bank.repos_on_amounts)
            )
            weighted_repo_maturity += list(
                np.array(bank.repos_off_maturities)
                * np.array(bank.repos_off_amounts)
            )
            total_repo_amount += sum(bank.repos_on_amounts) + sum(
                bank.repos_off_amounts
            )
        print(
            "Weighted average maturity of repos : {}".format(
                np.sum(weighted_repo_maturity) / total_repo_amount,
            )
        )

        print(
            "Mean of repos tot. volume {}".format(
                self.df_network_trajectory["Repos tot. volume"].mean()
            )
        )

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

        # Plot the reverse repo network
        binary_adj = np.where(
            self.adj_matrix > self.Network.min_repo_size, 1.0, 0.0
        )
        gx.plot_network(
            adj=self.adj_matrix,
            network_total_assets=self.Network.banks_total_assets,
            path=self.path_results + "repo_networks/",
            step=self.Network.step,
            name_in_title="reverse repo",
        )
        fct.save_np_array(
            self.adj_matrix,
            self.path_results + "repo_networks/adj_matrix",
        )

        # Plot the trust network
        gx.plot_network(
            adj=self.trust_adj_matrix.T / (self.trust_adj_matrix.std() + 1e-8),
            network_total_assets=self.Network.banks_total_assets,
            path=self.path_results + "trust_networks/",
            step=self.Network.step,
            name_in_title="trust",
        )
        fct.save_np_array(
            self.trust_adj_matrix,
            self.path_results + "trust_networks/trust",
        )

        # Plot the break-down of the balance per bank
        gx.bar_plot_balance_sheet(
            self.Network.banks_total_assets,
            self.dic_banks_assets,
            self.dic_banks_liabilities,
            self.dic_banks_off_balance,
            self.path_results + "balance_Sheets/",
            self.Network.step,
        )

        # Plot the break-down of the deposits per bank in relative shares
        gx.bar_plot_deposits(
            self.Network.banks_deposits,
            self.path_results + "deposits/",
            self.Network.step,
        )

        # Plot the core-periphery detection and assessment
        # special case here, an intermediary computation to keep track of p-values
        if self.cp_option:
            if self.Network.step > 0:
                bank_network = nx.from_numpy_matrix(
                    binary_adj, parallel_edges=False, create_using=nx.DiGraph
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
            self.Network.banks_total_assets,
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
        if self.Network.nb_steps % save_every != 0:
            self.store_trajectories()

        # final print
        self.final_print()

        if output_keys:
            output = self.build_output(output_keys)
            return output

    def save_param(self, save_every):
        with open(self.path_results + "param.txt", "w") as f:
            f.write(
                (
                    "n_banks={} \n"
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


def single_run(
    nb_banks=50,
    alpha_init=0.01,
    alpha=0.01,
    beta_init=0.1,
    beta_reg=0.1,
    beta_star=0.1,
    gamma=0.03,
    collateral_value=1.0,
    initialization_method="constant",
    alpha_pareto=1.3,
    shocks_method="bilateral",
    shocks_law="normal",
    shocks_vol=0.01,
    result_location="./results/",
    min_repo_size=1e-10,
    nb_steps=500,
    save_every=500,
    jaccard_periods=[20, 100, 250, 500],
    agg_periods=[20, 100, 250],
    cp_option=False,
    LCR_mgt_opt=True,
    output_keys=False,
):

    Network = ClassNetwork(
        nb_banks=nb_banks,
        alpha_init=alpha_init,
        beta_init=beta_init,
        beta_reg=beta_reg,
        beta_star=beta_star,
        alpha=alpha,
        gamma=gamma,
        collateral_value=collateral_value,
        initialization_method=initialization_method,
        alpha_pareto=alpha_pareto,
        shocks_method=shocks_method,
        shocks_law=shocks_law,
        shocks_vol=shocks_vol,
        min_repo_size=min_repo_size,
        LCR_mgt_opt=LCR_mgt_opt,
    )

    dynamics = ClassDynamics(
        Network,
        nb_steps=nb_steps,
        path_results=result_location,
        jaccard_periods=jaccard_periods,
        agg_periods=agg_periods,
        cp_option=cp_option,
    )

    output = dynamics.simulate(
        save_every=save_every,
        output_keys=output_keys,
    )

    return output
