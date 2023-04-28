import os

os.environ["OMP_NUM_THREADS"] = "1"

import networkx as nx
import numpy as np
from scipy.stats import pareto
from tqdm import tqdm
import graphics as gx
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
        self.df_bank_trajectory = pd.DataFrame(index=range(self.nb_steps))
        self.single_bank_id = 0  # the selected single bank id

        # total network trajectories
        self.df_network_trajectory = pd.DataFrame(index=range(self.nb_steps))

        # definition of the p-value parameter of the core-perihpery structure dected by the cpnet algo
        self.p_value = 1  # initialization at 1 (non significant test)

        self.dic_agg_binary_adj = (
            {}
        )  # dictionary of the aggregated ajency matrix over a given period
        for agg_period in self.agg_periods:
            self.dic_agg_binary_adj.update(
                {
                    agg_period: np.zeros(
                        (self.Network.nb_banks, self.Network.nb_banks)
                    )
                }
            )

        self.dic_prev_agg_binary_adj = (
            {}
        )  # dictionary of the previous aggregated binary adjency matrices for different aggregation periods
        for agg_period in self.agg_periods:
            self.dic_prev_agg_binary_adj.update(
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

    def step_record_trajectories(self):
        self.fill_df_network_trajectory()
        self.fill_df_bank_trajectory()

    def fill_df_network_trajectory(self):
        # accounting items
        for item in par.bank_items:
            self.df_network_trajectory.loc[
                self.Network.step, f"{item} tot. network"
            ] = self.Network.df_banks[item].sum()

        # degree
        bank_network = nx.from_numpy_array(
            self.Network.dic_matrices["binary_adjency"],
            parallel_edges=False,
            create_using=nx.DiGraph,
        )  # first define a networkx object.
        self.df_network_trajectory.loc[
            self.Network.step, "av. in-degree"
        ] = np.array(bank_network.in_degree())[:, 1].mean()

        # jaccard
        # first update agg adjancency matrix
        if self.Network.step > 0:
            for agg_period in self.agg_periods:
                if self.Network.step % agg_period > 0:
                    self.dic_agg_binary_adj.update(
                        {
                            agg_period: np.logical_or(
                                self.Network.dic_matrices["binary_adjency"],
                                self.dic_agg_binary_adj[agg_period],
                            )
                        }
                    )
                elif self.Network.step % agg_period == 0:
                    self.dic_agg_binary_adj.update(
                        {
                            agg_period: self.Network.dic_matrices[
                                "binary_adjency"
                            ]
                        }
                    )
        for agg_period in self.agg_periods:
            if self.Network.step == 0:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"jaccard index-{agg_period}",
                ] = 0

            elif self.Network.step % agg_period == agg_period - 1:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"jaccard index-{agg_period}",
                ] = (
                    np.logical_and(
                        self.dic_agg_binary_adj[agg_period],
                        self.dic_prev_agg_binary_adj[agg_period],
                    ).sum()
                    / np.logical_or(
                        self.dic_agg_binary_adj[agg_period],
                        self.dic_prev_agg_binary_adj[agg_period],
                    ).sum()
                )
                self.dic_prev_agg_binary_adj.update(
                    {agg_period: self.dic_agg_binary_adj[agg_period].copy()}
                )
            elif self.Network.step > 0:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"jaccard index-{agg_period}",
                ] = self.df_network_trajectory.loc[
                    self.Network.step - 1,
                    f"jaccard index-{agg_period}",
                ]

        # density
        for agg_period in self.agg_periods:
            if self.Network.step == 0:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"network density-{agg_period}",
                ] = 0
            elif self.Network.step % agg_period == agg_period - 1:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"network density-{agg_period}",
                ] = self.dic_agg_binary_adj[agg_period].sum() / (
                    self.Network.nb_banks * (self.Network.nb_banks - 1.0)
                )  # for a directed graph
            elif self.Network.step > 0:
                self.df_network_trajectory.loc[
                    self.Network.step,
                    f"network density-{agg_period}",
                ] = self.df_network_trajectory.loc[
                    self.Network.step - 1,
                    f"network density-{agg_period}",
                ]

        # gini
        self.df_network_trajectory.loc[self.Network.step, "gini"] = fct.gini(
            self.Network.df_banks["total assets"]
        )

        # repo exposures
        if len(self.Network.dic_matrices["non-zero_adjency"]) == 0:
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures min network"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures max network"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures av. network"
            ] = 0
        else:
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures min network"
            ] = np.min(self.Network.dic_matrices["non-zero_adjency"])
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures max network"
            ] = np.max(self.Network.dic_matrices["non-zero_adjency"])
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures av. network"
            ] = np.mean(self.Network.dic_matrices["non-zero_adjency"])

        if (
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures max network"
            ]
            > 0.99
        ):
            print("error")

        # Collateral reuse
        self.df_network_trajectory.loc[
            self.Network.step, "collateral reuse"
        ] = (
            self.df_network_trajectory.loc[
                self.Network.step, "securities reused tot. network"
            ]
        ) / (
            self.df_network_trajectory.loc[
                self.Network.step, "securities collateral tot. network"
            ]
            + 1e-10
        )

        # degree
        self.banks_degree = np.array(bank_network.degree())[:, 1]

    def fill_df_bank_trajectory(self):

        # Build the time series of the accounting items
        for item in par.bank_items:
            self.df_bank_trajectory.loc[
                self.Network.step, item
            ] = self.Network.df_banks.loc[self.single_bank_id, item]

        # In and Out-degree
        bank_network = nx.from_numpy_array(
            self.Network.dic_matrices["binary_adjency"],
            parallel_edges=False,
            create_using=nx.DiGraph,
        )
        self.df_bank_trajectory.loc[
            self.Network.step, "av. in-degree"
        ] = bank_network.in_degree(self.single_bank_id)
        self.df_bank_trajectory.loc[
            self.Network.step, "av. out-degree"
        ] = bank_network.out_degree(self.single_bank_id)

    def expost_record_trajectories(self):

        # build the df_reverse_repos history from all banks data
        self.Network.build_df_reverse_repos()

        # fill df_network_trajectory & df_bank_trajectory from the df_reverse_repos data
        for step in tqdm(
            range(self.Network.step + 1)
        ):  # +1 to cover all steps up to now
            self.expost_fill_df_network_trajectory(step)
            self.expost_fill_df_bank_trajectory(step)

    def expost_fill_df_network_trajectory(self, step):

        df = self.Network.df_reverse_repos

        # repos maturity av. network
        df_ending = df[df["maturity"] + df["start_step"] == step - 1]
        if df_ending["amount"].sum() > 0:
            self.df_network_trajectory.loc[
                step, "repos maturity av. network"
            ] = (df_ending["amount"] @ df_ending["maturity"]) / df_ending[
                "amount"
            ].sum()
        else:
            self.df_network_trajectory.loc[
                step, "repos maturity av. network"
            ] = 0

        # amount_ending_starting av. network
        df_starting = df[df["start_step"] == step - 1]
        nb_trans = len(df_ending) + len(df_starting)
        if nb_trans > 0:
            self.df_network_trajectory.loc[
                step,
                "amount_ending_starting av. network",
            ] = (
                df_ending["amount"].sum() + df_starting["amount"].sum()
            ) / nb_trans
        else:
            self.df_network_trajectory.loc[
                step,
                "amount_ending_starting av. network",
            ] = 0

    def expost_fill_df_bank_trajectory(self, step):

        df = self.Network.df_reverse_repos

        # check of the single bank entered into any reverse repos
        if self.single_bank_id in df.index.get_level_values(0):
            df = df.loc[self.single_bank_id]
        else:
            self.df_bank_trajectory.loc[step, "repos maturity av. bank"] = 0
            self.df_bank_trajectory.loc[
                step,
                "amount_ending_starting av. bank",
            ] = 0

        # repos maturity av. network
        df_ending = df[df["maturity"] + df["start_step"] == step - 1]
        if df_ending["amount"].sum() > 0:
            self.df_bank_trajectory.loc[step, "repos maturity av. bank"] = (
                df_ending["amount"] @ df_ending["maturity"]
            ) / df_ending["amount"].sum()
        else:
            self.df_bank_trajectory.loc[step, "repos maturity av. bank"] = 0

        # amount_ending_starting av. network
        df_starting = df[df["start_step"] == step - 1]
        nb_trans = len(df_ending) + len(df_starting)
        if nb_trans > 0:
            self.df_bank_trajectory.loc[
                step,
                "amount_ending_starting av. bank",
            ] = (
                df_ending["amount"].sum() + df_starting["amount"].sum()
            ) / nb_trans
        else:
            self.df_bank_trajectory.loc[
                step,
                "amount_ending_starting av. bank",
            ] = 0

        # nb_ending starting per signle bank
        self.df_bank_trajectory.loc[
            step,
            "nb_ending_starting",
        ] = nb_trans

    def plot_n_store_trajectories(self):
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
        self.Network.store_network(self.path_results)

        gx.plot_network(
            adj=self.Network.dic_matrices["adjency"],
            network_total_assets=self.Network.df_banks["total assets"],
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
            network_total_assets=self.Network.df_banks["total assets"],
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
            self.Network.df_banks["deposits"],
            self.path_results + "deposits/",
            self.Network.step,
        )

        # Plot the core-periphery detection and assessment
        # special case here, an intermediary computation to keep track of p-values
        if self.cp_option:
            if self.Network.step > 0:
                bank_network = nx.from_numpy_array(
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
                    name_in_title="reverse repo exposures",
                )  # plot charts

        # Plot the link between centrality and total asset size
        gx.plot_asset_per_degree(
            self.Network.df_banks["total assets"],
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

        # # Plot the time series of the average nb of transactions per step and per bank
        # gx.plot_average_nb_transactions(
        #     self.df_network_trajectory, self.path_results
        # )

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
        self.step_record_trajectories()
        self.expost_record_trajectories()
        self.plot_n_store_trajectories()

        # simulate the network
        for _ in tqdm(range(self.nb_steps - 1)):

            # run one step of the network
            self.Network.step_network()

            # record trajectories
            self.step_record_trajectories()

            # store only every given steps
            if self.Network.step % save_every == 0:
                self.expost_record_trajectories()
                self.plot_n_store_trajectories()

        # store the final step (if not already done)
        if self.nb_steps % save_every != 0:
            self.expost_record_trajectories()
            self.plot_n_store_trajectories()

        # final print
        self.print_summary()

        if output_keys:
            output = self.build_output(output_keys)
            return output

    def print_summary(self):
        for metric in [
            "repos maturity av. network",
            "repo exposures av. network",
        ]:
            print(
                f"{metric}:{self.df_network_trajectory.loc[self.Network.step, metric]}"
            )

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

        output.update({"core-peri. p-val.": self.p_value})

        return output
