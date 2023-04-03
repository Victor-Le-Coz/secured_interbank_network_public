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
        ], "Not valid initialisation method"

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
        self.network_deposits = np.zeros(
            n_banks
        )  # Numpy array of the deposits of
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
            self.agg_binary_adj_dic.update(
                {agg_period: np.zeros((n_banks, n_banks))}
            )

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

        # Specific case for the network density and jaccard aggregated
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
                "Shock doesn't sum to zero, sum is {}".format(
                    abs(shocks.sum())
                ),
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
            self.banks[
                i
            ].step_end_repos()  # Run the step end repos for the bank self

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

    def simulate(
        self, time_steps, save_every=10, output_opt=False, output_keys=None
    ):
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
                            (np.array(self.time_series_metrics[key]))[
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
