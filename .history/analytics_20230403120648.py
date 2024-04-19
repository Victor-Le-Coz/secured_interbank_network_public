

class ClassAnalytics:

    def __init__(self,time_series_metrics):
        self.time_series_metrics=



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
        for key in self.time_series_metrics.keys():
            self.time_series_metrics[key].append(0.0)

        # Loop over the banks and over the accounting items time series
        for i, bank in enumerate(self.banks):

            # Build the time series of the accounting items and store the
            # network dictionaries of the accounting items values
            for key in bank.assets.keys():  # only loop over assets items.
                self.time_series_metrics[key + " tot. volume"][
                    -1
                ] += bank.assets[
                    key
                ]  #
                # Computes the total of a given item at a given time step.
                self.network_assets[key][i] = bank.assets[key]  # Fill-in
                # the value of each accounting item of each bank into the
                # network asset dictionary.
            for key in bank.liabilities.keys():  # only loop over liabilities
                # items.
                self.time_series_metrics[key + " tot. volume"][
                    -1
                ] += bank.liabilities[key]
                self.network_liabilities[key][i] = bank.liabilities[key]
            for key in bank.off_balance.keys():  # only loop over off-balance
                # items.
                self.time_series_metrics[key + " tot. volume"][
                    -1
                ] += bank.off_balance[key]
                self.network_off_balance[key][i] = bank.off_balance[key]

            # Build the adjacency matrix of the reverse repos
            self.adj_matrix[i, :] = np.array(
                list(self.banks[i].reverse_repos.values())
            )

            # Build the adjacency matrix of the trust coefficients
            trusts = np.array(list(self.banks[i].trust.values()))
            self.trust_adj_matrix[i, :i] = trusts[:i]
            self.trust_adj_matrix[i, i + 1 :] = trusts[i:]

            # Build the total assets of each bank
            self.network_total_assets[i] = self.banks[i].total_assets()

            # Build the deposits numpy array of each bank
            self.network_deposits[i] = self.banks[i].liabilities["Deposits"]

            # Build the total network excess liquidity time series
            self.time_series_metrics["Excess Liquidity"][-1] += (
                self.banks[i].assets["Cash"]
                - self.banks[i].alpha * self.banks[i].liabilities["Deposits"]
            )

            # Build the weighted average maturity of repos (1/2).
            weighted_repo_maturity += list(
                np.array(self.banks[i].repos_on_maturities)
                * np.array(self.banks[i].repos_on_amounts)
            )  # add the on balance repos
            weighted_repo_maturity += list(
                np.array(self.banks[i].repos_off_maturities)
                * np.array(self.banks[i].repos_off_amounts)
            )
            total_repo_amount += sum(self.banks[i].repos_on_amounts) + sum(
                self.banks[i].repos_off_amounts
            )  # add the off balance repos

            # Build the time series of the Average number of repo transaction ended within a step (1/2).
            total_repo_transactions_counter += self.banks[
                i
            ].repo_transactions_counter  # compute the sum

            # Build the time series of the Average size of repo transaction ended within a step (1/2).
            total_repo_transactions_size += self.banks[
                i
            ].repo_transactions_size  # compute the sum

            # Build the time serie of the total assets across all banks
            self.time_series_metrics["Assets tot. volume"][
                -1
            ] += bank.total_assets()

        # clean the adj matrix from the negative values (otherwise the algo generate -1e-14 values for the reverse repos)
        self.adj_matrix[self.adj_matrix < 0] = 0

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(
            self.adj_matrix > self.min_repo_size, True, False
        )

        # build the aggregated adjancency matrix of the reverse repos at different aggregation periods
        if self.step > 0:
            for agg_period in self.agg_periods:
                if self.step % agg_period > 0:
                    self.agg_binary_adj_dic.update(
                        {
                            agg_period: np.logical_or(
                                binary_adj, self.agg_binary_adj_dic[agg_period]
                            )
                        }
                    )
                elif self.step % agg_period == 0:
                    self.agg_binary_adj_dic.update({agg_period: binary_adj})

        # Build the time series of the Av. nb. of repo transactions ended (2/2).
        self.time_series_metrics["Av. nb. of repo transactions ended"][-1] = (
            total_repo_transactions_counter / self.n_banks
        )

        # Build the time series of the Average volume of repo transaction ended within a step (2/2).
        if total_repo_transactions_counter != 0:
            self.time_series_metrics["Av. volume of repo transactions ended"][
                -1
            ] = (
                total_repo_transactions_size / total_repo_transactions_counter
            )
        else:
            self.time_series_metrics["Av. volume of repo transactions ended"][
                -1
            ] = 0

        # Build the time series of the weighted average maturity of the repo transactions (2/2)
        self.time_series_metrics["Repos av. maturity"][-1] = (
            np.sum(weighted_repo_maturity) / total_repo_amount
        )

        # Build the average in-degree in the network.
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )  # first define a networkx object.
        self.time_series_metrics["Av. in-degree"][-1] = np.array(
            bank_network.in_degree()
        )[:, 1].mean()

        # Build the jaccard index time series - version non aggregated.
        for jaccard_period in self.jaccard_periods:
            if self.step > 0 and self.step % jaccard_period == 0:

                self.time_series_metrics[
                    "Jaccard index " + str(jaccard_period) + " time steps"
                ][-1] = (
                    np.logical_and(
                        binary_adj, self.prev_binary_adj_dic[jaccard_period]
                    ).sum()
                    / np.logical_or(
                        binary_adj, self.prev_binary_adj_dic[jaccard_period]
                    ).sum()
                )
                self.prev_binary_adj_dic.update(
                    {jaccard_period: binary_adj.copy()}
                )
            elif self.step > 0:
                self.time_series_metrics[
                    "Jaccard index " + str(jaccard_period) + " time steps"
                ][-1] = self.time_series_metrics[
                    "Jaccard index " + str(jaccard_period) + " time steps"
                ][
                    -2
                ]

        # Build the jaccard index time series - version aggregated.
        for agg_period in self.agg_periods:
            if self.step % agg_period == agg_period - 1:
                self.time_series_metrics[
                    "Jaccard index over " + str(agg_period) + " time steps"
                ][-1] = (
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
            elif self.step > 0:
                self.time_series_metrics[
                    "Jaccard index over " + str(agg_period) + " time steps"
                ][-1] = self.time_series_metrics[
                    "Jaccard index over " + str(agg_period) + " time steps"
                ][
                    -2
                ]

        # Build the network density indicator.
        for agg_period in self.agg_periods:
            if self.step % agg_period == agg_period - 1:
                self.time_series_metrics[
                    "Network density over " + str(agg_period) + " time steps"
                ][-1] = self.agg_binary_adj_dic[agg_period].sum() / (
                    self.n_banks * (self.n_banks - 1.0)
                )  # for a directed graph
            elif self.step > 0:
                self.time_series_metrics[
                    "Network density over " + str(agg_period) + " time steps"
                ][-1] = self.time_series_metrics[
                    "Network density over " + str(agg_period) + " time steps"
                ][
                    -2
                ]

        # Build the gini coeficient of the network
        self.time_series_metrics["Gini"][-1] = fct.gini(
            self.network_total_assets
        )

        # Build the statistics regarding the size of the reverse repos across the network at a given time step
        non_zero_adj_matrix = self.adj_matrix[
            np.nonzero(self.adj_matrix)
        ]  # keep only non zero reverse repos

        if len(non_zero_adj_matrix) == 0:
            self.time_series_metrics["Repos min volume"][-1] = 0
            self.time_series_metrics["Repos max volume"][-1] = 0
            self.time_series_metrics["Repos av. volume"][-1] = 0
        else:
            self.time_series_metrics["Repos min volume"][-1] = np.min(
                non_zero_adj_matrix
            )
            self.time_series_metrics["Repos max volume"][-1] = np.max(
                non_zero_adj_matrix
            )
            self.time_series_metrics["Repos av. volume"][-1] = np.mean(
                non_zero_adj_matrix
            )

        # build the time serrie of Collateral reuse
        self.time_series_metrics["Collateral reuse"][-1] = (
            self.time_series_metrics["Securities Reused tot. volume"][-1]
        ) / (
            self.time_series_metrics["Securities Collateral tot. volume"][-1]
            + 1e-10
        )

        # Build the dictionary of the degree (total of in and out) of each node in the network at a given step
        self.network_degree = np.array(bank_network.degree())[:, 1]

    def comp_single_trajectory(self):

        # defin the single bank that we want to plot
        bank = self.banks[self.single_bank_id]

        # Initialization of each time serries (necessary to append a list)
        for key in self.single_trajectory.keys():
            self.single_trajectory[key].append(0.0)

        # Build the time series of the accounting item of the bank bank_id
        for key in bank.assets.keys():
            self.single_trajectory[key][-1] = bank.assets[key]
        for key in bank.liabilities.keys():
            self.single_trajectory[key][-1] = bank.liabilities[key]
        for key in bank.off_balance.keys():
            self.single_trajectory[key][-1] = bank.off_balance[key]

        # In and Out-degree
        binary_adj = np.where(
            self.adj_matrix > self.min_repo_size, True, False
        )
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )
        self.single_trajectory["Av. in-degree"][-1] = bank_network.in_degree(
            self.single_bank_id
        )
        self.single_trajectory["Av. out-degree"][-1] = bank_network.out_degree(
            self.single_bank_id
        )

        # Number of transactions of end repos per step
        self.single_trajectory["Nb. of repo transactions ended"][
            -1
        ] = self.banks[self.single_bank_id].repo_transactions_counter

        # size of transactions of end repos per step
        if self.banks[self.single_bank_id].repo_transactions_counter != 0:
            self.single_trajectory["Av. volume of repo transactions ended"][
                -1
            ] = (
                self.banks[self.single_bank_id].repo_transactions_size
                / self.banks[self.single_bank_id].repo_transactions_counter
            )
        else:
            self.single_trajectory["Av. volume of repo transactions ended"][
                -1
            ] = 0

        # Average across time of the weighted average maturity of repos
        self.single_trajectory["Repos av. maturity"][-1] = np.sum(
            list(
                np.array(self.banks[self.single_bank_id].repos_on_maturities)
                * np.array(self.banks[self.single_bank_id].repos_on_amounts)
            )
            + list(
                np.array(self.banks[self.single_bank_id].repos_off_maturities)
                * np.array(self.banks[self.single_bank_id].repos_off_amounts)
            )
        ) / (
            sum(self.banks[self.single_bank_id].repos_on_amounts)
            + sum(self.banks[self.single_bank_id].repos_off_amounts)
        )

    def comp_final_metrics(self):

        # Print the weighted average maturity of repos
        weighted_repo_maturity = []
        total_repo_amount = 0
        for i, bank in enumerate(self.banks):
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
                np.mean(self.time_series_metrics["Repos tot. volume"])
            )
        )

    def save_step_figures(self):
        """
        Instance method saving all the figures representing the network
        status at a given time-step as well as all the time series plots of the chosen metrics
        :return:
        """

        # Plot the reverse repo network
        binary_adj = np.where(self.adj_matrix > self.min_repo_size, 1.0, 0.0)
        gx.plot_network(
            adj=self.adj_matrix,
            network_total_assets=self.network_total_assets,
            path=self.result_location + "repo_networks/",
            step=self.step,
            name_in_title="reverse repo",
        )
        fct.save_np_array(
            self.adj_matrix, self.result_location + "repo_networks/adj_matrix"
        )

        # Plot the trust network
        gx.plot_network(
            adj=self.trust_adj_matrix.T / (self.trust_adj_matrix.std() + 1e-8),
            network_total_assets=self.network_total_assets,
            path=self.result_location + "trust_networks/",
            step=self.step,
            name_in_title="trust",
        )
        fct.save_np_array(
            self.trust_adj_matrix,
            self.result_location + "trust_networks/trust",
        )

        # Plot the break-down of the balance per bank
        gx.bar_plot_balance_sheet(
            self.network_total_assets,
            self.network_assets,
            self.network_liabilities,
            self.network_off_balance,
            self.result_location + "balance_Sheets/",
            self.step,
        )

        # Plot the break-down of the deposits per bank in relative shares
        gx.bar_plot_deposits(
            self.network_deposits,
            self.result_location + "deposits/",
            self.step,
        )

        # Plot the core-periphery detection and assessment
        # special case here, an intermediary computation to keep track of p-values
        if self.cp_option:
            if self.step > 0:
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
                    path=self.result_location + "core-periphery_structure/",
                    step=self.step,
                    name_in_title="reverse repos",
                )  # plot charts

        # Plot the link between centrality and total asset size
        gx.plot_asset_per_degree(
            self.network_total_assets,
            self.network_degree,
            self.result_location,
        )

        # Plot the time series of the total repos in the network
        gx.plot_repos(
            self.time_series_metrics,
            self.result_location,
        )

        # Plot the time series of the total MROS and loans in the network
        gx.plot_assets_loans_mros(
            self.time_series_metrics,
            self.result_location,
        )

        # Plot the time series of the securities usable, encumbered and
        # re-used in the network
        gx.plot_collateral(self.time_series_metrics, self.result_location)

        # Plot the time series of the weighted average number of time the
        # collateral is reused in the network
        gx.plot_collateral_reuse(
            self.time_series_metrics,
            self.result_location,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_not_aggregated(
            self.time_series_metrics,
            self.jaccard_periods,
            self.result_location,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_aggregated(
            self.time_series_metrics,
            self.agg_periods,
            self.result_location,
        )

        # Plot the time series of the total excess liquidity and deposits in
        # the network
        gx.plot_excess_liquidity_and_deposits(
            self.time_series_metrics, self.result_location
        )

        # Plot the time series of the network density
        gx.plot_network_density(
            self.time_series_metrics, self.agg_periods, self.result_location
        )

        # Plot the time series of the gini coefficients
        gx.plot_gini(self.time_series_metrics, self.result_location)

        # Plot the time series of the statistics of the size of reverse repo
        gx.plot_reverse_repo_size_stats(
            self.time_series_metrics, self.result_location
        )

        # Plot the time series of the network average degree
        gx.plot_degre_network(self.time_series_metrics, self.result_location)

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_nb_transactions(
            self.time_series_metrics, self.result_location
        )

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_size_transactions(
            self.time_series_metrics, self.result_location
        )

        # Plot the average maturity of repos.
        gx.plot_average_maturity_repo(
            self.time_series_metrics, self.result_location
        )

        # Plot the single bank trajectory time series.
        gx.plot_single_trajectory(self.single_trajectory, self.result_location)


