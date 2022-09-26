import os
import shutil

import networkx as nx
import numpy as np
from scipy.stats import pareto
from tqdm import tqdm

import graphics as gx
from bank import ClassBank
import shocks as sh
import function as fct


class ClassNetwork:
    """
    This class models a network of ClassBank instances.
    """

    def __init__(
        self,
        n_banks,
        alpha=1.0,
        beta_init=1,
        beta_reg=10.0,
        beta_star=10.0,
        gamma=3.0,
        collateral_value=1.0,
        initialization_method="constant",
        alpha_pareto=0.5,
        shock_method="dirichlet",
        shocks_vol=0.1,
        result_location="./results/",
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
        assert shock_method in [
            "bilateral",
            "multilateral",
            "dirichlet",
            "non-conservative",
        ], "Not valid initialisation method :"

        # initialization of the class parameters.
        self.n_banks = n_banks
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_reg = beta_reg
        self.beta_star = beta_star
        self.gamma = gamma
        self.collateral_value = collateral_value
        self.initialization_method = initialization_method
        self.alpha_pareto = alpha_pareto
        self.shock_method = shock_method
        if shock_method == "non-conservative":
            self.conservative_shock = False
        else:
            self.conservative_shock = True
        self.shocks_vol = shocks_vol
        self.result_location = result_location

        # Definition of the internal parameters of the ClassNetwork.
        self.steps = 0  # Step number in the simulation process
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
        self.prev_adj_matrix = np.zeros((n_banks, n_banks))  # previous
        # adjacency matrix, used for the computation of the jaccard index (
        # stable trading relationships)

        # Definition of the dictionary associating to each accounting item the list of its values across time for a single bank. It also includes other time serries metrics, like the excess liquidity the in-degree, the out-degree, the nb of repos transactions ended within a step and the average across time of the maturity of repos.
        self.single_trajectory = {}
        self.single_bank_id = 0  # the selected single bank id

        # Definition of the dictionary associating to each accounting item,
        # the list of its total value across time. It also includes other
        # time series metrics, like the network density, the jaccard index,
        # or the excess liquidity.
        self.time_series_metrics = {}

        # Reset the network when creating an instance of the ClassNetwork
        self.reset_network()

    def reset_network(self):
        """
        Instance method allowing the initialization an instance of the
        ClassNetwork. It notably sets the starting deposits according to the
        chosen approach.
        :return:
        """

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
                    * 100.0
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
                    alpha=self.alpha,
                    beta_init=self.beta_init,
                    beta_reg=self.beta_reg,
                    beta_star=self.beta_star,
                    gamma=self.gamma,
                    collateral_value=self.collateral_value,
                    conservative_shock=self.conservative_shock,
                )
            )

        # For loop of the banks in the network to initialize each of the
        # banks parameters of the instances of ClassBank. The banks
        # parameter in the ClassBank is a dictionary of the instances of the
        # ClassBank class existing in the ClassNetwork class, while the
        # banks parameter in the ClassNetwork is a list.
        for bank in self.banks:
            bank.initialize_banks(self.banks)

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
            "In-degree": [],
            "Out-degree": [],
            "Number of repo transaction ended within a step": [],
            "Size of repo transaction ended within a step": [],
            "Maturity of repos": [],
        }

        # Initialize the other network level and aggregated level parameters
        self.time_series_metrics = {
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
            "In-degree": [],
            "Excess Liquidity": [],
            "Jaccard Index": [],
            "Network Density": [],
            "Average number of repo transaction ended within a step": [],
            "Average size of repo transaction ended within a step": [],
            "Average maturity of repos": [],
            "Gini": [],
        }
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
        self.steps = 0.0

        # Create the required path to store the results
        if os.path.exists(self.result_location):  # Delete all previous figures
            shutil.rmtree(self.result_location)
        os.makedirs(os.path.join(self.result_location, "Reverse_repo_Networks"))
        os.makedirs(os.path.join(self.result_location, "Trust_Networks"))
        os.makedirs(os.path.join(self.result_location, "Core-periphery_structure"))
        os.makedirs(os.path.join(self.result_location, "Deposits"))
        os.makedirs(os.path.join(self.result_location, "BalanceSheets"))
        os.makedirs(os.path.join(self.result_location, "Assets_per_degree"))

        # Update all the metrics at time step 0
        self.compute_step_metrics()
        self.compute_single_trajectory()

    def step_network(self):
        """
        Instance method allowing the computation of the next step status of
        the network.
        :return:
        """

        # Generation of the shocks
        if self.shock_method == "bilateral":
            shocks = sh.generate_bilateral_shocks(
                self.network_deposits, law="uniform", vol=self.shocks_vol
            )
        elif self.shock_method == "multilateral":  # Damien's proposal,
            # doesn't work yet, could be enhanced
            shocks = sh.generate_multilateral_shocks(
                self.network_deposits, law="uniform", vol=self.shocks_vol
            )
        elif self.shock_method == "dirichlet":
            shocks = sh.generate_dirichlet_shocks(
                self.network_deposits,
                self.network_initial_deposits,
                option="mean-reverting",
                vol=self.shocks_vol,
            )
        elif self.shock_method == "non-conservative":
            shocks = sh.generate_non_conservative_shocks(
                self.network_deposits, law="log-normal", vol=self.shocks_vol
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
        for i in ix:
            self.banks[i].assert_minimum_reserves()
            self.banks[i].assert_alm()
            self.banks[i].assert_lcr()
            # self.banks[i].assert_leverage()
            self.banks[i].steps += 1

    def simulate(self, time_steps, save_every=10, jaccard_period=10):
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
        self.jaccard_period = jaccard_period
        for _ in tqdm(range(time_steps)):
            if self.steps % save_every == 0.0:
                self.save_step_figures()
            self.step_network()
            self.compute_step_metrics()
            self.compute_single_trajectory()
            self.steps += 1
        # for bank in self.banks:
        #     print(bank)
        #     print(
        #         "Bank Deposits {} Bank Cash {}".format(
        #             bank.liabilities["Deposits"], bank.assets["Cash"]
        #         )
        #     )
        self.save_step_figures()
        self.compute_final_metrics()

    # <editor-fold desc="Metrics updates, saving, and printing">
    def compute_step_metrics(self):
        """
        Instance method allowing the computation of the time_series_metrics
        parameter as well as the network_assets, network_liabilities and
        network_off_balance dictionaries.
        :return:
        """
        # initialization of the list used to compute the weighted average repo maturity
        weighted_repo_maturity = []
        total_repo_amount = 0

        # initialization of the counter of the repos transactions ended within a step across all banks
        total_repo_transactions_counter = 0

        # initialization of the total amount of the repos transaction ended ended within a step across all banks
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
                self.time_series_metrics[key][-1] += bank.assets[key]  #
                # Computes the total of a given item at a given time step.
                self.network_assets[key][i] = bank.assets[key]  # Fill-in
                # the value of each accounting item of each bank into the
                # network asset dictionary.
            for key in bank.liabilities.keys():  # only loop over liabilities
                # items.
                self.time_series_metrics[key][-1] += bank.liabilities[key]
                self.network_liabilities[key][i] = bank.liabilities[key]
            for key in bank.off_balance.keys():  # only loop over off-balance
                # items.
                self.time_series_metrics[key][-1] += bank.off_balance[key]
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

        # Build the time series of the Average number of repo transaction ended within a step (2/2).
        self.time_series_metrics[
            "Average number of repo transaction ended within a step"
        ][-1] = (total_repo_transactions_counter / self.n_banks)

        # Build the time series of the Average number of repo transaction ended within a step (2/2).
        if total_repo_transactions_counter != 0:
            self.time_series_metrics[
                "Average size of repo transaction ended within a step"
            ][-1] = (total_repo_transactions_size / total_repo_transactions_counter)
        else:
            self.time_series_metrics[
                "Average size of repo transaction ended within a step"
            ][-1] = 0

        # Build the time series of the weighted average maturity of the repo transactions (2/2)
        self.time_series_metrics["Average maturity of repos"][-1] = (
            np.sum(weighted_repo_maturity) / total_repo_amount
        )

        # Build the average in-degree in the network.
        binary_adj = np.where(self.adj_matrix > 0.0, True, False)
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )  # first define a networkx object.
        self.time_series_metrics["In-degree"][-1] = np.array(bank_network.in_degree())[
            :, 1
        ].mean()

        # Build the jaccard index time series.
        prev_binary_adj = np.where(self.prev_adj_matrix > 0.0, True, False)
        if self.steps > 0 and self.steps % self.jaccard_period == 0:
            self.time_series_metrics["Jaccard Index"][-1] = (
                np.logical_and(binary_adj, prev_binary_adj).sum()
                / np.logical_or(binary_adj, prev_binary_adj).sum()
            )
            self.prev_adj_matrix = self.adj_matrix.copy()
        elif self.steps > 0:
            self.time_series_metrics["Jaccard Index"][-1] = self.time_series_metrics[
                "Jaccard Index"
            ][-2]

        # Build the network density indicator.
        self.time_series_metrics["Network Density"][-1] = (
            2.0 * binary_adj.sum() / (self.n_banks * (self.n_banks - 1.0))
        )

        # Build the gini coeficient of the network
        self.time_series_metrics["Gini"][-1] = fct.gini(self.network_total_assets)

        # Build the dictionary of the degree (total of in and out) of each node in the network at a given step
        self.network_degree = np.array(bank_network.degree())[:, 1]

        # Build the single trajectory time serries of a given bank
        self.banks[0]

    def compute_single_trajectory(self):

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
        binary_adj = np.where(self.adj_matrix > 0.0, True, False)
        bank_network = nx.from_numpy_matrix(
            binary_adj,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )
        self.single_trajectory["In-degree"][-1] = bank_network.in_degree(
            self.single_bank_id
        )
        self.single_trajectory["Out-degree"][-1] = bank_network.out_degree(
            self.single_bank_id
        )

        # Number of transactions of end repos per step
        self.single_trajectory["Number of repo transaction ended within a step"][
            -1
        ] = self.banks[self.single_bank_id].repo_transactions_counter

        # size of transactions of end repos per step
        if self.banks[self.single_bank_id].repo_transactions_counter != 0:
            self.single_trajectory["Size of repo transaction ended within a step"][
                -1
            ] = (
                self.banks[self.single_bank_id].repo_transactions_size
                / self.banks[self.single_bank_id].repo_transactions_counter
            )
        else:
            self.single_trajectory["Size of repo transaction ended within a step"][
                -1
            ] = 0

        # Average across time of the weighted average maturity of repos
        self.single_trajectory["Maturity of repos"][-1] = np.sum(
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

    def compute_final_metrics(self):

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
            "Average amount of repos {}".format(
                np.mean(self.time_series_metrics["Repos"])
            )
        )

    def save_step_figures(self):
        """
        Instance method saving all the figures representing the network
        status at a given time-step as well as all the time series plots of the chosen metrics
        :return:
        """

        # Plot the reverse repo network
        binary_adj = np.where(self.adj_matrix > 0.0, 1.0, 0.0)
        gx.plot_network(
            self.adj_matrix,
            os.path.join(self.result_location, "Reverse_Repo_Networks"),
            self.steps,
            "Reverse_Repo",
        )

        # Plot the trust network
        gx.plot_network(
            self.trust_adj_matrix.T / (self.trust_adj_matrix.std() + 1e-8),
            os.path.join(self.result_location, "Trust_Networks"),
            self.steps,
            "Trust",
        )

        # Plot the break-down of the balance per bank
        gx.bar_plot_balance_sheet(
            self.network_total_assets,
            self.network_assets,
            self.network_liabilities,
            self.network_off_balance,
            os.path.join(self.result_location, "BalanceSheets"),
            self.steps,
        )

        # Plot the break-down of the deposits per bank in relative shares
        gx.bar_plot_deposits(
            self.network_deposits,
            os.path.join(self.result_location, "Deposits"),
            self.steps,
        )

        # Plot the core-periphery detection and assessment
        gx.plot_core_periphery(
            self.adj_matrix,
            os.path.join(self.result_location, "Core-periphery_structure"),
            self.steps,
            "Repos",
        )

        # Plot the link between centrality and total asset size
        gx.plot_asset_per_degree(
            self.network_total_assets,
            self.network_degree,
            os.path.join(self.result_location, "Assets_per_degree"),
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
            np.array(self.time_series_metrics["Securities Reused"])
            / (np.array(self.time_series_metrics["Securities Collateral"]) + 1e-8),
            self.result_location,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_not_aggregated(
            self.time_series_metrics,
            self.jaccard_period,
            self.result_location,
        )

        # Plot the time series of the jaccard index
        gx.plot_jaccard_aggregated(
            self.time_series_metrics,
            self.jaccard_period,
            self.result_location,
        )

        # Plot the time series of the total excess liquidity and deposits in
        # the network
        gx.plot_excess_liquidity_and_deposits(
            self.time_series_metrics, self.result_location
        )

        # Plot the time series of the network density
        gx.plot_network_density(self.time_series_metrics, self.result_location)

        # Plot the time series of the gini coefficients
        gx.plot_gini(self.time_series_metrics, self.result_location)

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
