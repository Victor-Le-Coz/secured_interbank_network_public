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


class ClassNetwork:
    """
    This class models a network of ClassBank instances.
    """

    def __init__(
        self,
        n_banks,
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
        LCR_mgt_opt=True,
        jaccard_periods=[20, 100, 250],
        agg_periods=[20, 100, 250],
        cp_option=False,
    ):
        """
        Instance methode initializing the ClassNetwork.
        :param n_banks: number of instances of the ClassBank in the network.
        :param alpha: the share of deposits required as minimum reserves (
        currently 1% in the Eurozone).
        :param beta_init: the initial LCR share of deposits used to define
        the amount of securities usable.
        :param beta_reg: the regulatory LCR share of deposits required (
        currently 10% in the Eurozone).
        :param beta_star: the targeted LCR  share of deposits => could be
        modelled, currently set as a constant.
        :param gamma: the share of total asset required as minimum leverage
        ratio (currently 3% in the Eurozone).
        :param collateral_value: value of the collateral in monetary units
        => could be modelled, currently set as constant.
        :param initialization_method: method of initialization of the
        deposits' distribution across the network of banks, either constant
        or pareto.
        :param alpha_pareto: coefficient of the pareto law used within the
        initialization method.
        :param shock_method: generation method for the input shocks on the
        deposits.
        :param shocks_vol: volatility of the input shocks on the deposits.
        :param result_location: location path for the folder storing the
        resulting plots.
        """

        # Tests to ensure the adequate parameters were chosen when creating
        # the network
        assert initialization_method in ["constant", "pareto"], (
            "Not valid initialisation method :" " 'constant' or 'pareto'"
        )
        assert shocks_method in [
            "bilateral",
            "multilateral",
            "dirichlet",
            "non-conservative",
        ], "Not valid initialisation method :"

        # initialization of the class parameters.
        self.n_banks = n_banks
        self.alpha_init = alpha_init
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_reg = beta_reg
        self.beta_star = beta_star
        self.gamma = gamma
        self.collateral_value = collateral_value
        self.initialization_method = initialization_method
        self.alpha_pareto = alpha_pareto
        self.shocks_method = shocks_method
        if shocks_method == "non-conservative":
            self.conservative_shock = False
        else:
            self.conservative_shock = True
        self.shocks_law = shocks_law
        self.shocks_vol = shocks_vol
        self.result_location = result_location
        self.min_repo_size = min_repo_size
        self.LCR_mgt_opt = LCR_mgt_opt
        self.jaccard_periods = jaccard_periods
        self.agg_periods = agg_periods
        self.cp_option = cp_option

        # Definition of the internal parameters of the ClassNetwork.
        self.step = 0  # Step number in the simulation process
        self.banks = []  # List of the instances of the ClassBank existing
        # in the ClassNetwork.
        self.network_deposits = np.zeros(n_banks)  # Numpy array of the deposits of
        # the banks in the network.
        self.network_initial_deposits = np.zeros(n_banks)  # Numpy array of the
        # initial deposits of the banks in the network.
        self.network_total_assets = np.zeros(n_banks)  # Numpy array of the
        # total assets of the banks in the network.
        self.network_degree = np.zeros(
            n_banks
        )  # Numpy array of the degree of the banks in the network.

        # Definition of the dictionaries associating to each of the accounting
        # items, its corresponding numpy array of its value per bank,
        # at a given time step.
        self.network_assets = {}
        self.network_liabilities = {}
        self.network_off_balance = {}

        # Definition of the value of the collateral, here a constant across
        # time
        self.collateral_value = 1.0

        # Definition of the adjacency matrices
        self.adj_matrix = np.zeros((n_banks, n_banks))  # reverse repos
        # adjacency matrix
        self.trust_adj_matrix = np.zeros((n_banks, n_banks))  # trust
        # coeficients adjacency matrix
        self.prev_binary_adj_dic = (
            {}
        )  # dictionary of the previous adjacency matrix, used for the computation of the jaccard index (stable trading relationships) of different time length
        for jaccard_period in jaccard_periods:
            self.prev_binary_adj_dic.update(
                {jaccard_period: np.zeros((n_banks, n_banks))}
            )

        self.agg_binary_adj_dic = (
            {}
        )  # dictionary of the aggregated ajency matrix over a given period
        for agg_period in agg_periods:
            self.agg_binary_adj_dic.update({agg_period: np.zeros((n_banks, n_banks))})

        self.prev_agg_binary_adj_dic = (
            {}
        )  # dictionary of the previous aggregated binary adjency matrices for different aggregation periods
        for agg_period in agg_periods:
            self.prev_agg_binary_adj_dic.update(
                {agg_period: np.zeros((n_banks, n_banks))}
            )

        # Definition of the dictionary associating to each accounting item the list of its values across time for a single bank. It also includes other time serries metrics, like the excess liquidity the in-degree, the out-degree, the nb of repo transactions ended within a step and the average across time of the maturity of repos.
        self.single_trajectory = {}
        self.single_bank_id = 0  # the selected single bank id

        # Definition of the dictionary associating to each accounting item,
        # the list of its total value across time. It also includes other
        # time series metrics, like the network density, the jaccard index,
        # or the excess liquidity.
        self.time_series_metrics = {}

        # definition of the p-value parameter of the core-perihpery structure dected by the cpnet algo
        self.p_value = 1  # initialization at 1 (non significant test)

        # Reset the network when creating an instance of the ClassNetwork
        self.reset_network()

    def reset_network(self):
        """
        Instance method allowing the initialization an instance of the
        ClassNetwork. It notably sets the starting deposits according to the
        chosen approach.
        :return:
        """

        # firt: reset the banks parameter of the instance of ClassNetwork
        self.banks = []

        # For loop over the number of banks in the network to build the
        # deposits and initial deposits numpy arrays according to the chosen
        # method, then create each of the instances of ClassBank
        for b in range(self.n_banks):
            if self.initialization_method == "pareto":
                deposits = (
                    pareto.rvs(
                        self.alpha_pareto,
                        loc=0,
                        scale=1,
                        size=1,
                        random_state=None,
                    )[0]
                    * 40.0
                )
            elif self.initialization_method == "constant":
                deposits = 100.0
            else:
                assert False, ""
            self.network_deposits[b] = deposits
            self.network_initial_deposits[b] = deposits
            self.banks.append(
                ClassBank(
                    id=b,
                    initial_deposits=deposits,
                    alpha_init=self.alpha_init,
                    alpha=self.alpha,
                    beta_init=self.beta_init,
                    beta_reg=self.beta_reg,
                    beta_star=self.beta_star,
                    gamma=self.gamma,
                    collateral_value=self.collateral_value,
                    conservative_shock=self.conservative_shock,
                    LCR_mgt_opt=self.LCR_mgt_opt,
                )
            )

        # For loop of the banks in the network to initialize each of the
        # banks parameters of the instances of ClassBank. The banks
        # parameter in the ClassBank is a dictionary of the instances of the
        # ClassBank class existing in the ClassNetwork class, while the
        # banks parameter in the ClassNetwork is a list.
        for bank in self.banks:
            bank.initialize_banks(self.banks)

        # initialize other single trajectory metrics
        self.single_trajectory = {
            "Cash": [],
            "Securities Usable": [],
            "Securities Encumbered": [],
            "Loans": [],
            "Reverse Repos": [],
            "Own Funds": [],
            "Deposits": [],
            "Repos": [],
            "MROs": [],
            "Securities Collateral": [],
            "Securities Reused": [],
            "Excess Liquidity": [],
            "Av. in-degree": [],
            "Av. out-degree": [],
            "Nb. of repo transactions ended": [],
            "Av. volume of repo transactions ended": [],
            "Repos av. maturity": [],
        }

        # Initialize the other network level and aggregated level parameters
        self.time_series_metrics = {
            "Cash tot. volume": [],
            "Securities Usable tot. volume": [],
            "Securities Encumbered tot. volume": [],
            "Loans tot. volume": [],
            "Reverse Repos tot. volume": [],
            "Own Funds tot. volume": [],
            "Deposits tot. volume": [],
            "Repos tot. volume": [],
            "MROs tot. volume": [],
            "Securities Collateral tot. volume": [],
            "Securities Reused tot. volume": [],
            "Av. in-degree": [],
            "Excess Liquidity": [],
            "Av. nb. of repo transactions ended": [],
            "Av. volume of repo transactions ended": [],
            "Repos av. maturity": [],
            "Gini": [],
            "Repos min volume": [],
            "Repos max volume": [],
            "Repos av. volume": [],
            "Assets tot. volume": [],
            "Collateral reuse": [],
        }

        # Specific case of the Jaccard periods
        for jaccard_period in self.jaccard_periods:
            self.time_series_metrics.update(
                {"Jaccard index " + str(jaccard_period) + " time steps": []}
            )

        # Specific case for the network density
        for agg_period in self.agg_periods:
            self.time_series_metrics.update(
                {"Network density over " + str(agg_period) + " time steps": []}
            )
            self.time_series_metrics.update(
                {"Jaccard index over " + str(agg_period) + " time steps": []}
            )

        self.network_liabilities = {
            "Own Funds": np.zeros(self.n_banks),
            "Deposits": np.zeros(self.n_banks),
            "Repos": np.zeros(self.n_banks),
            "MROs": np.zeros(self.n_banks),
        }
        self.network_assets = {
            "Cash": np.zeros(self.n_banks),
            "Securities Usable": np.zeros(self.n_banks),
            "Securities Encumbered": np.zeros(self.n_banks),
            "Loans": np.zeros(self.n_banks),
            "Reverse Repos": np.zeros(self.n_banks),
        }
        self.network_off_balance = {
            "Securities Collateral": np.zeros(self.n_banks),
            "Securities Reused": np.zeros(self.n_banks),
        }

        # Initialize the steps to 0
        self.step = 0.0

        # Create the required path to store the results
        fct.init_results_path(self.result_location)

        # Update all the metrics at time step 0
        self.comp_step_metrics()
        self.comp_single_trajectory()

    def step_network(self):
        """
        Instance method allowing the computation of the next step status of
        the network.
        :return:
        """

        # Generation of the shocks
        if self.shocks_method == "bilateral":
            shocks = sh.generate_bilateral_shocks(
                self.network_deposits, law=self.shocks_law, vol=self.shocks_vol
            )
        elif self.shocks_method == "multilateral":  # Damien's proposal,
            # doesn't work yet, could be enhanced
            shocks = sh.generate_multilateral_shocks(
                self.network_deposits, law=self.shocks_law, vol=self.shocks_vol
            )
        elif self.shocks_method == "dirichlet":
            shocks = sh.generate_dirichlet_shocks(
                self.network_deposits,
                self.network_initial_deposits,
                option="mean-reverting",
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "non-conservative":
            shocks = sh.generate_non_conservative_shocks(
                self.network_deposits,
                self.network_initial_deposits,
                self.network_total_assets,
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        else:
            assert False, ""

        # Tests to ensure the shock created matches the required properties
        assert (
            np.min(self.network_deposits + shocks) >= 0
        ), "negative shocks larger than deposits"  # To ensure shocks are not
        # higher than the deposits amount of each bank
        if self.conservative_shock:
            assert (
                abs(shocks.sum()) == 0.0,
                "Shock doesn't sum to zero, sum is {}".format(abs(shocks.sum())),
            )  # To ensure that the conservative shocks are dully conservative

        # For loops over the instances of ClassBank in the ClassNetwork.
        ix = np.arange(self.n_banks)  # Defines an index of the banks
        for i in ix:
            self.banks[i].set_shock(shocks[i])
            self.banks[i].set_collateral(self.collateral_value)
            if self.LCR_mgt_opt:
                self.banks[i].step_lcr_mgt()
            self.banks[
                i
            ].repo_transactions_counter = (
                0  # Reset the repo transaction ended counter to 0
            )
            self.banks[
                i
            ].repo_transactions_size = (
                0  # Reset the repo transaction ended counter to 0
            )
        ix = np.random.permutation(ix)  # Permutation of the
        # banks' indexes to decide in which order banks can close their repos.
        for i in ix:
            self.banks[i].step_end_repos()  # Run the step end repos for the bank self

        ix = np.random.permutation(ix)  # New permutation of the
        # banks' indexes to decide in which order banks can enter into repos
        for i in ix:
            self.banks[i].step_enter_repos()
            if not (self.conservative_shock) or not (self.LCR_mgt_opt):
                self.banks[i].step_MRO()
        for i in ix:
            self.banks[i].assert_minimum_reserves()
            self.banks[i].assert_alm()
            if self.LCR_mgt_opt:
                self.banks[i].assert_lcr()
            # self.banks[i].assert_leverage()
            self.banks[i].steps += 1

    def simulate(self, time_steps, save_every=10, output_opt=False, output_keys=None):
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
            if self.step % save_every == 0.0:
                self.save_step_figures()
            self.step_network()
            self.comp_step_metrics()
            self.comp_single_trajectory()
            self.step += 1
        # for bank in self.banks:
        #     print(bank)
        #     print(
        #         "Bank Deposits {} Bank Cash {}".format(
        #             bank.liabilities["Deposits"], bank.assets["Cash"]
        #         )
        #     )
        self.save_step_figures()
        self.comp_final_metrics()

        # build output
        if output_opt:
            output = self.build_output(output_keys)
            return output

    # <editor-fold desc="Metrics updates, saving, and printing">
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
                self.time_series_metrics[key + " tot. volume"][-1] += bank.assets[
                    key
                ]  #
                # Computes the total of a given item at a given time step.
                self.network_assets[key][i] = bank.assets[key]  # Fill-in
                # the value of each accounting item of each bank into the
                # network asset dictionary.
            for key in bank.liabilities.keys():  # only loop over liabilities
                # items.
                self.time_series_metrics[key + " tot. volume"][-1] += bank.liabilities[
                    key
                ]
                self.network_liabilities[key][i] = bank.liabilities[key]
            for key in bank.off_balance.keys():  # only loop over off-balance
                # items.
                self.time_series_metrics[key + " tot. volume"][-1] += bank.off_balance[
                    key
                ]
                self.network_off_balance[key][i] = bank.off_balance[key]

            # Build the adjacency matrix of the reverse repos
            self.adj_matrix[i, :] = np.array(list(self.banks[i].reverse_repos.values()))

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
            self.time_series_metrics["Assets tot. volume"][-1] += bank.total_assets()

        # clean the adj matrix from the negative values (otherwise the algo generate -1e-14 values for the reverse repos)
        self.adj_matrix[self.adj_matrix < 0] = 0

        # build a binary adjency matrix from the weighted adjency matrix
        binary_adj = np.where(self.adj_matrix > self.min_repo_size, True, False)

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
            self.time_series_metrics["Av. volume of repo transactions ended"][-1] = (
                total_repo_transactions_size / total_repo_transactions_counter
            )
        else:
            self.time_series_metrics["Av. volume of repo transactions ended"][-1] = 0

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
                self.prev_binary_adj_dic.update({jaccard_period: binary_adj.copy()})
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
        self.time_series_metrics["Gini"][-1] = fct.gini(self.network_total_assets)

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
        ) / (self.time_series_metrics["Securities Collateral tot. volume"][-1] + 1e-10)

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
        binary_adj = np.where(self.adj_matrix > self.min_repo_size, True, False)
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
        self.single_trajectory["Nb. of repo transactions ended"][-1] = self.banks[
            self.single_bank_id
        ].repo_transactions_counter

        # size of transactions of end repos per step
        if self.banks[self.single_bank_id].repo_transactions_counter != 0:
            self.single_trajectory["Av. volume of repo transactions ended"][-1] = (
                self.banks[self.single_bank_id].repo_transactions_size
                / self.banks[self.single_bank_id].repo_transactions_counter
            )
        else:
            self.single_trajectory["Av. volume of repo transactions ended"][-1] = 0

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
                np.array(bank.repos_on_maturities) * np.array(bank.repos_on_amounts)
            )
            weighted_repo_maturity += list(
                np.array(bank.repos_off_maturities) * np.array(bank.repos_off_amounts)
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
            self.trust_adj_matrix, self.result_location + "trust_networks/trust"
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
        gx.plot_reverse_repo_size_stats(self.time_series_metrics, self.result_location)

        # Plot the time series of the network average degree
        gx.plot_degre_network(self.time_series_metrics, self.result_location)

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_nb_transactions(self.time_series_metrics, self.result_location)

        # Plot the time series of the average nb of transactions per step and per bank
        gx.plot_average_size_transactions(
            self.time_series_metrics, self.result_location
        )

        # Plot the average maturity of repos.
        gx.plot_average_maturity_repo(self.time_series_metrics, self.result_location)

        # Plot the single bank trajectory time series.
        gx.plot_single_trajectory(self.single_trajectory, self.result_location)

    # </editor-fold>

    def save_param(self, time_steps, save_every):
        with open(self.result_location + "param.txt", "w") as f:
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
                    self.n_banks,
                    self.alpha,
                    self.beta_init,
                    self.beta_reg,
                    self.beta_star,
                    self.gamma,
                    self.collateral_value,
                    self.initialization_method,
                    self.alpha_pareto,
                    self.shocks_method,
                    self.shocks_law,
                    self.shocks_vol,
                    self.result_location,
                    self.min_repo_size,
                    time_steps,
                    save_every,
                    self.jaccard_periods,
                    self.LCR_mgt_opt,
                )
            )

    def build_output(self, output_keys):
        output = {}
        stat_len_step = 250

        # build the time series metrics outputs
        for key in output_keys:

            # handeling specific cases
            if key == "Core-Peri. p_val.":
                output.update({"Core-Peri. p_val.": self.p_value})

            elif key == "Jaccard index":
                for jaccard_period in self.jaccard_periods:
                    output.update(
                        {
                            "Jaccard index "
                            + str(jaccard_period)
                            + " time steps": np.mean(
                                (
                                    np.array(
                                        self.time_series_metrics[
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
                for agg_period in self.agg_periods:
                    output.update(
                        {
                            key
                            + str(agg_period)
                            + " time steps": np.mean(
                                (
                                    np.array(
                                        self.time_series_metrics[
                                            key + str(agg_period) + " time steps"
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
                            (np.array(self.time_series_metrics[key]))[-stat_len_step:]
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
    output_opt=False,
    LCR_mgt_opt=True,
    output_keys=None,
):

    network = ClassNetwork(
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

    if output_opt:
        return network.simulate(
            time_steps=time_steps,
            save_every=save_every,
            output_opt=output_opt,
            output_keys=output_keys,
        )

    else:
        network.simulate(
            time_steps=time_steps,
            save_every=save_every,
            output_opt=output_opt,
            output_keys=output_keys,
        )
