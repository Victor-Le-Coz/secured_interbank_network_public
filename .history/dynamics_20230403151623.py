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


class ClassDynamics:
    def __init__(self, Network):
        self.Network = Network

    def comp_step_metrics(self):
        """
        Instance method allowing the computation of the time_series_metrics
        parameter as well as the network_assets, network_liabilities and
        network_off_balance dictionaries.
        :return:
        """
        # initialization of the list used to compute the weighted average repo maturity
        weighted_repo_maturity = []
        total_repo_amount = 0

        # initialization of the counter of the repo transactions ended within a step across all banks
        total_repo_transactions_counter = 0

        # initialization of the total amount of the repo transactions ended ended within a step across all banks
        total_repo_transactions_size = 0

        # Add the first item 0 to each of the time series, it is necessary
        # to allow to append a list with list[-1] => not optimal however !
        for key in self.Network.time_series_metrics.keys():
            self.Network.time_series_metrics[key].append(0.0)

        # Loop over the banks and over the accounting items time series
        for i, bank in enumerate(self.Network.banks):

            # Build the time series of the accounting items and store the
            # network dictionaries of the accounting items values
            for key in bank.assets.keys():  # only loop over assets items.
                self.Network.time_series_metrics[key + " tot. volume"][
                    -1
                ] += bank.assets[
                    key
                ]  #
                # Computes the total of a given item at a given time step.
                self.Network.network_assets[key][i] = bank.assets[
                    key
                ]  # Fill-in
                # the value of each accounting item of each bank into the
                # network asset dictionary.
            for key in bank.liabilities.keys():  # only loop over liabilities
                # items.
                self.Network.time_series_metrics[key + " tot. volume"][
                    -1
                ] += bank.liabilities[key]
                self.Network.network_liabilities[key][i] = bank.liabilities[
                    key
                ]
            for key in bank.off_balance.keys():  # only loop over off-balance
                # items.
                self.Network.time_series_metrics[key + " tot. volume"][
                    -1
                ] += bank.off_balance[key]
                self.Network.network_off_balance[key][i] = bank.off_balance[
                    key
                ]

            # Build the adjacency matrix of the reverse repos
            self.Network.adj_matrix[i, :] = np.array(
                list(self.Network.banks[i].reverse_repos.values())
            )

            # Build the adjacency matrix of the trust coefficients
            trusts = np.array(list(self.Network.banks[i].trust.values()))
            self.Network.trust_adj_matrix[i, :i] = trusts[:i]
            self.Network.trust_adj_matrix[i, i + 1 :] = trusts[i:]

            # Build the total assets of each bank
            self.Network.network_total_assets[i] = self.Network.banks[
                i
            ].total_assets()

            # Build the deposits numpy array of each bank
            self.Network.network_deposits[i] = self.Network.banks[
                i
            ].liabilities["Deposits"]

            # Build the total network excess liquidity time series
            self.Network.time_series_metrics["Excess Liquidity"][-1] += (
                self.Network.banks[i].assets["Cash"]
                - self.Network.banks[i].alpha
                * self.Network.banks[i].liabilities["Deposits"]
            )

            # Build the weighted average maturity of repos (1/2).
            weighted_repo_maturity += list(
                np.array(self.Network.banks[i].repos_on_maturities)
                * np.array(self.Network.banks[i].repos_on_amounts)
            )  # add the on balance repos
            weighted_repo_maturity += list(
                np.array(self.Network.banks[i].repos_off_maturities)
                * np.array(self.Network.banks[i].repos_off_amounts)
            )
            total_repo_amount += sum(
                self.Network.banks[i].repos_on_amounts
            ) + sum(
                self.Network.banks[i].repos_off_amounts
            )  # add the off balance repos

            # Build the time series of the Average number of repo transaction ended within a step (1/2).
            total_repo_transactions_counter += self.Network.banks[
                i
            ].repo_transactions_counter  # compute the sum

            # Build the time series of the Average size of repo transaction ended within a step (1/2).
            total_repo_transactions_size += self.Network.banks[
                i
            ].repo_transactions_size  # compute the sum

            # Build the time serie of the total assets across all banks
            self.Network.time_series_metrics["Assets tot. volume"][
                -1
            ] += bank.total_assets()

        # clean the adj matrix from the negative values (otherwise the algo generate -1e-14 values for the reverse repos)
        self.Network.adj_matrix[self.Network.adj_matrix < 0] = 0

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(
            self.Network.adj_matrix > self.Network.min_repo_size, True, False
        )

        # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
        if self.Network.step > 0:
            for agg_period in self.Network.agg_periods:
                if self.Network.step % agg_period > 0:
                    self.Network.agg_binary_adj_dic.update(
                        {
                            agg_period: np.logical_or(
                                binary_adj,
                                self.Network.agg_binary_adj_dic[agg_period],
                            )
                        }
                    )
                elif self.Network.step % agg_period == 0:
                    self.Network.agg_binary_adj_dic.update(
                        {agg_period: binary_adj}
                    )

        # Build the time series of the Av. nb. of repo transactions ended (2/2).
        self.Network.time_series_metrics["Av. nb. of repo transactions ended"][
            -1
        ] = (total_repo_transactions_counter / self.Network.n_banks)

        # Build the time series of the Average volume of repo transaction ended within a step (2/2).
        if total_repo_transactions_counter != 0:
            self.Network.time_series_metrics[
                "Av. volume of repo transactions ended"
            ][-1] = (
                total_repo_transactions_size / total_repo_transactions_counter
            )
        else:
            self.Network.time_series_metrics[
                "Av. volume of repo transactions ended"
            ][-1] = 0

        # Build the time series of the weighted average maturity of the repo transactions (2/2)
        self.Network.time_series_metrics["Repos av. maturity"][-1] = (
            np.sum(weighted_repo_maturity) / total_repo_amount
        )

        # Build the average in-degree in the network.
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )  # first define a networkx object.
        self.Network.time_series_metrics["Av. in-degree"][-1] = np.array(
            bank_network.in_degree()
        )[:, 1].mean()

        # Build the jaccard index time series - version non aggregated.
        for jaccard_period in self.Network.jaccard_periods:
            if (
                self.Network.step > 0
                and self.Network.step % jaccard_period == 0
            ):

                self.Network.time_series_metrics[
                    "Jaccard index " + str(jaccard_period) + " time steps"
                ][-1] = (
                    np.logical_and(
                        binary_adj,
                        self.Network.prev_binary_adj_dic[jaccard_period],
                    ).sum()
                    / np.logical_or(
                        binary_adj,
                        self.Network.prev_binary_adj_dic[jaccard_period],
                    ).sum()
                )
                self.Network.prev_binary_adj_dic.update(
                    {jaccard_period: binary_adj.copy()}
                )
            elif self.Network.step > 0:
                self.Network.time_series_metrics[
                    "Jaccard index " + str(jaccard_period) + " time steps"
                ][-1] = self.Network.time_series_metrics[
                    "Jaccard index " + str(jaccard_period) + " time steps"
                ][
                    -2
                ]

        # Build the jaccard index time series - version aggregated.
        for agg_period in self.Network.agg_periods:
            if self.Network.step % agg_period == agg_period - 1:
                self.Network.time_series_metrics[
                    "Jaccard index over " + str(agg_period) + " time steps"
                ][-1] = (
                    np.logical_and(
                        self.Network.agg_binary_adj_dic[agg_period],
                        self.Network.prev_agg_binary_adj_dic[agg_period],
                    ).sum()
                    / np.logical_or(
                        self.Network.agg_binary_adj_dic[agg_period],
                        self.Network.prev_agg_binary_adj_dic[agg_period],
                    ).sum()
                )
                self.Network.prev_agg_binary_adj_dic.update(
                    {
                        agg_period: self.Network.agg_binary_adj_dic[
                            agg_period
                        ].copy()
                    }
                )
            elif self.Network.step > 0:
                self.Network.time_series_metrics[
                    "Jaccard index over " + str(agg_period) + " time steps"
                ][-1] = self.Network.time_series_metrics[
                    "Jaccard index over " + str(agg_period) + " time steps"
                ][
                    -2
                ]

        # Build the network density indicator.
        for agg_period in self.Network.agg_periods:
            if self.Network.step % agg_period == agg_period - 1:
                self.Network.time_series_metrics[
                    "Network density over " + str(agg_period) + " time steps"
                ][-1] = self.Network.agg_binary_adj_dic[agg_period].sum() / (
                    self.Network.n_banks * (self.Network.n_banks - 1.0)
                )  # for a directed graph
            elif self.Network.step > 0:
                self.Network.time_series_metrics[
                    "Network density over " + str(agg_period) + " time steps"
                ][-1] = self.Network.time_series_metrics[
                    "Network density over " + str(agg_period) + " time steps"
                ][
                    -2
                ]

        # Build the gini coeficient of the network
        self.Network.time_series_metrics["Gini"][-1] = fct.gini(
            self.Network.network_total_assets
        )

        # Build the statistics regarding the size of the reverse repos across the network at a given time step
        non_zero_adj_matrix = self.Network.adj_matrix[
            np.nonzero(self.Network.adj_matrix)
        ]  # keep only non zero reverse repos

        if len(non_zero_adj_matrix) == 0:
            self.Network.time_series_metrics["Repos min volume"][-1] = 0
            self.Network.time_series_metrics["Repos max volume"][-1] = 0
            self.Network.time_series_metrics["Repos av. volume"][-1] = 0
        else:
            self.Network.time_series_metrics["Repos min volume"][-1] = np.min(
                non_zero_adj_matrix
            )
            self.Network.time_series_metrics["Repos max volume"][-1] = np.max(
                non_zero_adj_matrix
            )
            self.Network.time_series_metrics["Repos av. volume"][-1] = np.mean(
                non_zero_adj_matrix
            )

        # build the time serrie of Collateral reuse
        self.Network.time_series_metrics["Collateral reuse"][-1] = (
            self.Network.time_series_metrics["Securities Reused tot. volume"][
                -1
            ]
        ) / (
            self.Network.time_series_metrics[
                "Securities Collateral tot. volume"
            ][-1]
            + 1e-10
        )

        # Build the dictionary of the degree (total of in and out) of each node in the network at a given step
        self.Network.network_degree = np.array(bank_network.degree())[:, 1]

    def comp_single_trajectory(self):

        # defin the single bank that we want to plot
        bank = self.Network.banks[self.Network.single_bank_id]

        # Initialization of each time serries (necessary to append a list)
        for key in self.Network.single_trajectory.keys():
            self.Network.single_trajectory[key].append(0.0)

        # Build the time series of the accounting item of the bank bank_id
        for key in bank.assets.keys():
            self.Network.single_trajectory[key][-1] = bank.assets[key]
        for key in bank.liabilities.keys():
            self.Network.single_trajectory[key][-1] = bank.liabilities[key]
        for key in bank.off_balance.keys():
            self.Network.single_trajectory[key][-1] = bank.off_balance[key]

        # In and Out-degree
        binary_adj = np.where(
            self.Network.adj_matrix > self.Network.min_repo_size, True, False
        )
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )
        self.Network.single_trajectory["Av. in-degree"][
            -1
        ] = bank_network.in_degree(self.Network.single_bank_id)
        self.Network.single_trajectory["Av. out-degree"][
            -1
        ] = bank_network.out_degree(self.Network.single_bank_id)

        # Number of transactions of end repos per step
        self.Network.single_trajectory["Nb. of repo transactions ended"][
            -1
        ] = self.Network.banks[
            self.Network.single_bank_id
        ].repo_transactions_counter

        # size of transactions of end repos per step
        if (
            self.Network.banks[
                self.Network.single_bank_id
            ].repo_transactions_counter
            != 0
        ):
            self.Network.single_trajectory[
                "Av. volume of repo transactions ended"
            ][-1] = (
                self.Network.banks[
                    self.Network.single_bank_id
                ].repo_transactions_size
                / self.Network.banks[
                    self.Network.single_bank_id
                ].repo_transactions_counter
            )
        else:
            self.Network.single_trajectory[
                "Av. volume of repo transactions ended"
            ][-1] = 0

        # Average across time of the weighted average maturity of repos
        self.Network.single_trajectory["Repos av. maturity"][-1] = np.sum(
            list(
                np.array(
                    self.Network.banks[
                        self.Network.single_bank_id
                    ].repos_on_maturities
                )
                * np.array(
                    self.Network.banks[
                        self.Network.single_bank_id
                    ].repos_on_amounts
                )
            )
            + list(
                np.array(
                    self.Network.banks[
                        self.Network.single_bank_id
                    ].repos_off_maturities
                )
                * np.array(
                    self.Network.banks[
                        self.Network.single_bank_id
                    ].repos_off_amounts
                )
            )
        ) / (
            sum(
                self.Network.banks[
                    self.Network.single_bank_id
                ].repos_on_amounts
            )
            + sum(
                self.Network.banks[
                    self.Network.single_bank_id
                ].repos_off_amounts
            )
        )

    def comp_final_metrics(self):

        # Print the weighted average maturity of repos
        weighted_repo_maturity = []
        total_repo_amount = 0
        for i, bank in enumerate(self.Network.banks):
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
                np.mean(self.Network.time_series_metrics["Repos tot. volume"])
            )
        )

    def save_step_figures(self):
        """
        Instance method saving all the figures representing the network
        status at a given time-step as well as all the time series plots of the chosen metrics
        :return:
        """

        # Plot the reverse repo network
        binary_adj = np.where(
            self.Network.adj_matrix > self.Network.min_repo_size, 1.0, 0.0
        )
        gx.plot_network(
            adj=self.Network.adj_matrix,
            network_total_assets=self.Network.network_total_assets,
            path=self.Network.result_location + "repo_networks/",
            step=self.Network.step,
            name_in_title="reverse repo",
        )
        fct.save_np_array(
            self.Network.adj_matrix,
            self.Network.result_location + "repo_networks/adj_matrix",
        )

        # Plot the trust network
        gx.plot_network(
            adj=self.Network.trust_adj_matrix.T
            / (self.Network.trust_adj_matrix.std() + 1e-8),
            network_total_assets=self.Network.network_total_assets,
            path=self.Network.result_location + "trust_networks/",
            step=self.Network.step,
            name_in_title="trust",
        )
        fct.save_np_array(
            self.Network.trust_adj_matrix,
            self.Network.result_location + "trust_networks/trust",
        )

        # Plot the break-down of the balance per bank
        gx.bar_plot_balance_sheet(
            self.Network.network_total_assets,
            self.Network.network_assets,
            self.Network.network_liabilities,
            self.Network.network_off_balance,
            self.Network.result_location + "balance_Sheets/",
            self.Network.step,
        )

        # Plot the break-down of the deposits per bank in relative shares
        gx.bar_plot_deposits(
            self.Network.network_deposits,
            self.Network.result_location + "deposits/",
            self.Network.step,
        )

        # Plot the core-periphery detection and assessment
        # special case here, an intermediary computation to keep track of p-values
        if self.Network.cp_option:
            if self.Network.step > 0:
                bank_network = nx.from_numpy_matrix(
                    binary_adj, parallel_edges=False, create_using=nx.DiGraph
                )  # build nx object
                sig_c, sig_x, significant, p_value = fct.cpnet_test(
                    bank_network
                )  # run cpnet test
                self.Network.p_value = p_value  # record p_value
                gx.plot_core_periphery(
                    bank_network=bank_network,
                    sig_c=sig_c,
                    sig_x=sig_x,
                    path=self.Network.result_location
                    + "core-periphery_structure/",
                    step=self.Network.step,
                    name_in_title="reverse repos",
                )  # plot charts

        # Plot the link between centrality and total asset size
        gx.plot_asset_per_degree(
            self.Network.network_total_assets,
            self.Network.network_degree,
            self.Network.result_location,
        )

        # Plot the time series of the total repos in the network
        gx.plot_repos(
            self.Network.time_series_metrics,
            self.Network.result_location,
        )

        # Plot the time series of the total MROS and loans in the network
        gx.plot_assets_loans_mros(
            self.Network.time_series_metrics,
            self.Network.result_location,
        )

        # Plot the time series of the securities usable, encumbered and
        # re-used in the network
        gx.plot_collateral(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the time series of the weighted average number of time the
        # collateral is reused in the network
        gx.plot_collateral_reuse(
            self.Network.time_series_metrics,
            self.Network.result_location,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_not_aggregated(
            self.Network.time_series_metrics,
            self.Network.jaccard_periods,
            self.Network.result_location,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_aggregated(
            self.Network.time_series_metrics,
            self.Network.agg_periods,
            self.Network.result_location,
        )

        # Plot the time series of the total excess liquidity and deposits in
        # the network
        gx.plot_excess_liquidity_and_deposits(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the time series of the network density
        gx.plot_network_density(
            self.Network.time_series_metrics,
            self.Network.agg_periods,
            self.Network.result_location,
        )

        # Plot the time series of the gini coefficients
        gx.plot_gini(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the time series of the statistics of the size of reverse repo
        gx.plot_reverse_repo_size_stats(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the time series of the network average degree
        gx.plot_degre_network(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_nb_transactions(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_size_transactions(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the average maturity of repos.
        gx.plot_average_maturity_repo(
            self.Network.time_series_metrics, self.Network.result_location
        )

        # Plot the single bank trajectory time series.
        gx.plot_single_trajectory(
            self.Network.single_trajectory, self.Network.result_location
        )

    def simulate(self, time_steps, save_every=10, output_keys=None):
        """
        Instance method for the simulation of the ABM.
        :param time_steps: number of time_steps of the simulation, could be
        seen as a number of days, given that collateral calls back must be
        met inside a given time step.
        :param save_every: frequency of the saving of the plots, except for
        the time series that are recorded at all steps.
        :param jaccard_period: period over which the jaccard index is computed.
        :return:
        """
        self.save_param(time_steps, save_every)
        for _ in tqdm(range(time_steps)):
            if self.Network.step % save_every == 0.0:
                # Update all the metrics at time step 0
                self.comp_step_metrics()
                self.comp_single_trajectory()
                self.save_step_figures()
            self.Network.step_network()
            self.comp_step_metrics()
            self.comp_single_trajectory()
            self.Network.step += 1
        self.save_step_figures()
        self.comp_final_metrics()

        output = self.build_output(output_keys)
        return output

    def save_param(self, time_steps, save_every):
        with open(self.Network.result_location + "param.txt", "w") as f:
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
                    "time_steps={} \n"
                    "save_every={} \n"
                    "jaccard_periods={} \n"
                    "LCR_mgt_opt={} \n"
                ).format(
                    self.Network.n_banks,
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
                    self.Network.result_location,
                    self.Network.min_repo_size,
                    time_steps,
                    save_every,
                    self.Network.jaccard_periods,
                    self.Network.LCR_mgt_opt,
                )
            )

    def build_output(self, output_keys):
        output = {}
        stat_len_step = 250

        # build the time series metrics outputs
        for key in output_keys:

            # handeling specific cases
            if key == "Core-Peri. p_val.":
                output.update({"Core-Peri. p_val.": self.Network.p_value})

            elif key == "Jaccard index":
                for jaccard_period in self.Network.jaccard_periods:
                    output.update(
                        {
                            "Jaccard index "
                            + str(jaccard_period)
                            + " time steps": np.mean(
                                (
                                    np.array(
                                        self.Network.time_series_metrics[
                                            "Jaccard index "
                                            + str(jaccard_period)
                                            + " time steps"
                                        ]
                                    )
                                )[-stat_len_step:]
                            )
                        }
                    )

            elif key in ["Jaccard index over ", "Network density over "]:
                for agg_period in self.Network.agg_periods:
                    output.update(
                        {
                            key
                            + str(agg_period)
                            + " time steps": np.mean(
                                (
                                    np.array(
                                        self.Network.time_series_metrics[
                                            key
                                            + str(agg_period)
                                            + " time steps"
                                        ]
                                    )
                                )[-stat_len_step:]
                            )
                        }
                    )

            else:
                output.update(
                    {
                        key: np.mean(
                            (np.array(self.Network.time_series_metrics[key]))[
                                -stat_len_step:
                            ]
                        )
                    }
                )

        return output


def single_run(
    n_banks=50,
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
    time_steps=500,
    save_every=500,
    jaccard_periods=[20, 100, 250, 500],
    agg_periods=[20, 100, 250],
    cp_option=False,
    LCR_mgt_opt=True,
    output_keys=None,
):

    Network = ClassNetwork(
        n_banks=n_banks,
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
        result_location=result_location,
        min_repo_size=min_repo_size,
        LCR_mgt_opt=LCR_mgt_opt,
        jaccard_periods=jaccard_periods,
        agg_periods=agg_periods,
        cp_option=cp_option,
    )

    dynamics = ClassDynamics(Network)

    return dynamics.simulate(
        time_steps=time_steps,
        save_every=save_every,
        output_keys=output_keys,
    )
