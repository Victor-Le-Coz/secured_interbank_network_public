import numpy as np
from scipy.stats import pareto
from bank import ClassBank
from tqdm import tqdm
from scipy.stats import truncnorm
import networkx as nx
import os
import shutil
import graphics as gx

float_limit = 1e-8


class ClassNetwork:
    def __init__(
        self,
        n_banks,
        alpha_pareto=0.5,
        beta_init=1,
        beta_reg=10.0,
        beta_star=10.0,
        initial_mr=1.0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
        shock_method="log-normal",
        std_law=0.1,
        result_location="./results/",
    ):
        assert init in ["constant", "pareto"], (
            "Not valid initialisation method :" " 'constant' or 'pareto'"
        )
        assert shock_method in ["log-normal", "dirichlet", "conservative"], (
            "Not valid initialisation method :"
            " 'log-normal' or 'dirichlet' or 'conservative'"
        )
        # Params
        self.n_banks = n_banks
        self.alpha_pareto = alpha_pareto
        self.beta_init = beta_init
        self.beta_reg = beta_reg
        self.beta_star = beta_star
        self.alpha = initial_mr
        self.gamma = initial_l2s
        self.collateral_value = collateral_value
        self.init = init
        self.shock_method = shock_method
        if shock_method == "dirichlet":
            self.std_control = 1.0 / (std_law ** 2.0)
            self.conservative_shock = True
        elif shock_method == "conservative":
            self.conservative_shock = True
        elif shock_method == "log-normal":
            self.std_control = np.sqrt(np.log(1.0 + std_law ** 2.0))
            self.conservative_shock = False
        elif shock_method == "normal":
            self.std_control = std_law
            self.conservative_shock = False
        else:
            self.std_control = std_law
            self.conservative_shock = False
        self.result_location = result_location

        # Internal
        self.banks = []
        self.deposits = np.zeros(n_banks)
        self.balance_sheets = np.zeros(n_banks)
        self.total_assets = {}
        self.total_liabilities = {}
        self.total_off_balance = {}
        self.collateral = 1.0
        self.adj_matrix = np.zeros((n_banks, n_banks))
        self.trust_adj_matrix = np.zeros((n_banks, n_banks))
        self.prev_adj_matrix = np.zeros((n_banks, n_banks))
        self.metrics = {}
        self.total_steps = 0.0
        self.init_depotits = np.zeros(n_banks)
        self.reset_network()

    def reset_network(self):
        for b in range(self.n_banks):
            if self.init == "pareto":
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
            elif self.init == "constant":
                deposits = 100.0
            else:
                assert False, ""
            self.deposits[b - 1] = deposits
            self.init_depotits[b - 1] = deposits
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
        for bank in self.banks:
            bank.initialize_banks(self.banks)
        self.metrics = {
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
            "Degree": [],
            "Excess Liquidity": [],
            "Jaccard Index": [],
            "Network Density": [],
        }
        self.total_liabilities = {
            "Own Funds": np.zeros(self.n_banks),
            "Deposits": np.zeros(self.n_banks),
            "Repos": np.zeros(self.n_banks),
            "MROs": np.zeros(self.n_banks),
        }
        self.total_assets = {
            "Cash": np.zeros(self.n_banks),
            "Securities Usable": np.zeros(self.n_banks),
            "Securities Encumbered": np.zeros(self.n_banks),
            "Loans": np.zeros(self.n_banks),
            "Reverse Repos": np.zeros(self.n_banks),
        }
        self.total_off_balance = {
            "Securities Collateral": np.zeros(self.n_banks),
            "Securities Reused": np.zeros(self.n_banks),
        }
        self.total_steps = 0.0
        if os.path.exists(self.result_location):
            shutil.rmtree(self.result_location)
        os.makedirs(os.path.join(self.result_location, "Repo_Networks"))
        os.makedirs(os.path.join(self.result_location, "Trust_Networks"))
        os.makedirs(os.path.join(self.result_location, "Deposits"))
        os.makedirs(os.path.join(self.result_location, "BalanceSheets"))
        self.update_metrics()

    def update_metrics(self):
        bank_network = nx.from_numpy_matrix(
            self.adj_matrix, parallel_edges=False, create_using=nx.DiGraph,
        )
        for key in self.metrics.keys():
            self.metrics[key].append(0.0)
        for i, bank in enumerate(self.banks):
            for key in bank.assets.keys():
                self.metrics[key][-1] += bank.assets[key]
                self.total_assets[key][i] = bank.assets[key]
            for key in bank.liabilities.keys():
                self.metrics[key][-1] += bank.liabilities[key]
                self.total_liabilities[key][i] = bank.liabilities[key]
            for key in bank.off_balance.keys():
                self.metrics[key][-1] += bank.off_balance[key]
                self.total_off_balance[key][i] = bank.off_balance[key]
            self.adj_matrix[i, :] = np.array(
                list(self.banks[i].reverse_repos.values())
            )
            trusts = np.array(list(self.banks[i].trust.values()))
            self.trust_adj_matrix[i, :i] = trusts[:i]
            self.trust_adj_matrix[i, i + 1 :] = trusts[i:]
            self.balance_sheets[i] = self.banks[i].total_assets()
            self.deposits[i] = self.banks[i].liabilities["Deposits"]
            self.metrics["Excess Liquidity"][-1] += (
                self.banks[i].assets["Cash"]
                - self.banks[i].alpha * self.banks[i].liabilities["Deposits"]
            )
        self.metrics["Degree"][-1] = np.array(bank_network.in_degree())[
            :, 1
        ].mean()
        binary_adj = np.where(self.adj_matrix > 0.0, True, False)
        prev_binary_adj = np.where(self.prev_adj_matrix > 0.0, True, False)
        if self.total_steps > 0 and self.total_steps % self.period == 0:
            self.metrics["Jaccard Index"][-1] = (
                np.logical_and(binary_adj, prev_binary_adj).sum()
                / np.logical_or(binary_adj, prev_binary_adj).sum()
            )
            self.prev_adj_matrix = self.adj_matrix.copy()
        elif self.total_steps > 0:
            self.metrics["Jaccard Index"][-1] = self.metrics["Jaccard Index"][
                -2
            ]
        self.metrics["Network Density"][-1] = (
            2.0 * binary_adj.sum() / (self.n_banks * (self.n_banks - 1.0))
        )

    def final_metrics(self):

        # duration
        weighted_durations = []
        total_amount = 0
        for i, bank in enumerate(self.banks):
            weighted_durations += list(
                np.array(bank.repos_on_maturities)
                * np.array(bank.repos_on_amounts)
            )
            weighted_durations += list(
                np.array(bank.repos_off_maturities)
                * np.array(bank.repos_off_amounts)
            )
            total_amount += sum(bank.repos_on_amounts) + sum(
                bank.repos_off_amounts
            )
        print(
            "Weighted average maturity of repos : {}".format(
                np.sum(weighted_durations) / total_amount,
            )
        )

        print(
            "Average amount of repos {}".format(np.mean(self.metrics["Repos"]))
        )

    def save_figs(self):
        gx.plot_network(
            self.adj_matrix / (self.adj_matrix.std() + 1e-8),
            os.path.join(self.result_location, "Repo_Networks"),
            self.total_steps,
            "Repos",
        )

        gx.plot_network(
            self.trust_adj_matrix.T / (self.trust_adj_matrix.std() + 1e-8),
            os.path.join(self.result_location, "Trust_Networks"),
            self.total_steps,
            "Trust",
        )

        gx.bar_plot_balance_sheet(
            self.balance_sheets,
            self.total_assets,
            self.total_liabilities,
            self.total_off_balance,
            os.path.join(self.result_location, "BalanceSheets"),
            self.total_steps,
        )

        gx.bar_plot_deposits(
            self.deposits,
            os.path.join(self.result_location, "Deposits"),
            self.total_steps,
        )

    def save_time_series(self):
        gx.plot_repos(
            self.metrics, self.result_location,
        )
        gx.plot_loans_mro(
            self.metrics, self.result_location,
        )
        gx.plot_collateral(self.metrics, self.result_location)
        gx.plot_jaccard(self.metrics, self.period, self.result_location)
        gx.plot_excess_liquidity_and_deposits(
            self.metrics, self.result_location
        )
        gx.plot_network_density(self.metrics, self.result_location)
        gx.plot_collateral_reuse(
            np.array(self.metrics["Securities Reused"])
            / (np.array(self.metrics["Securities Collateral"]) + 1e-8),
            self.result_location,
        )

    def step_network(self):
        if self.shock_method == "conservative":
            # define middle of the liste of banks
            N_max = (
                self.n_banks - self.n_banks % 2
            )  # can not apply a shock on one bank if odd nb
            N_half = int(self.n_banks / 2)

            # create a permutation of all the deposits amounts
            ix = np.arange(self.n_banks)  # create an index
            ix_p = np.random.permutation(ix)  # permutation of the index
            deposits_p = self.deposits[
                ix_p
            ]  # define the permuted array of deposits

            # apply a negative relative shock on the first half of the banks
            rho_neg = np.random.uniform(
                -1, 0, size=N_half
            )  # case uniform  law
            # rho_neg = -np.random.beta(1, 1, size=N_half)  # case beta  law

            # apply a positive relative shock on the second half of the banks
            rho_pos = (
                -rho_neg * deposits_p[0:N_half] / deposits_p[N_half:N_max]
            )

            # concatenate the relative shocks
            if self.n_banks > N_max:
                rho = np.concatenate([rho_neg, rho_pos, [0]])
            elif self.n_banks == N_max:
                rho = np.concatenate([rho_neg, rho_pos])

            # build an un-permuted array of absolute shocks
            shocks = np.zeros(self.n_banks)

            # compute the absolute shock from the deposit amount
            shocks[ix_p] = deposits_p * rho

            assert (
                abs(shocks.sum()) < float_limit,
                "Shock doesn't sum to zero, sum is {}".format(
                    abs(shocks.sum())
                ),
            )
            assert (
                self.deposits + shocks
            ).all() > -float_limit, "negative shocks larger than deposits"
        elif self.shock_method == "dirichlet":
            # dirichlet approach
            deposits = self.deposits + 1e-8
            # dispatch = np.random.dirichlet(
            #     (deposits / deposits.sum()) * self.constant_dirichlet
            # )
            # deposits = self.deposits.sum() * dispatch
            # dispatch = np.random.dirichlet(
            #     (self.init_depotits / self.init_depotits.sum())
            #     * self.std_control
            # )
            dispatch = np.random.dirichlet(
                np.ones(self.n_banks) / self.n_banks * self.std_control
            )
            deposits = self.deposits.sum() * dispatch
            shocks = deposits - self.deposits
            assert abs(shocks.sum()) < float_limit, "Shock doesn't sum to zero"
        elif self.shock_method == "log-normal":
            # log-normal approach
            deposits = (
                np.random.lognormal(
                    mean=-0.5 * self.std_control ** 2,
                    sigma=self.std_control,
                    size=len(self.banks),
                )
                * self.deposits
            )
            shocks = deposits - self.deposits
        elif self.shock_method == "normal":
            # Lux's approach but with truncated gaussian
            # shocks = (
            #     truncnorm.rvs(-3, 3, size=len(self.banks)) * self.deposits / 3
            # )
            deposits = np.maximum(
                self.deposits
                + np.random.randn(self.n_banks) * self.std_control,
                0.0,
            )
            shocks = deposits - self.deposits
        else:
            assert False, ""

        # print("Minimum shock is :", min(self.deposits + shocks))
        # print("Sum of shocks is {}".format(round(shocks.sum(), 2)))
        # print("Shocks : ", shocks)
        # assert abs(shocks.sum()) < 1e-8, "Shock doesn't sum to zero"

        ix = np.arange(self.n_banks)
        for i in ix:
            self.banks[i].set_shock(shocks[i])
            self.banks[i].set_collateral(self.collateral)
            self.banks[i].step_lcr_mgt()

        ix = np.random.permutation(ix)
        for i in ix:
            self.banks[i].step_end_repos()
        ix = np.random.permutation(ix)
        for i in ix:
            self.banks[i].step_enter_repos()
        for i in ix:
            self.banks[i].assert_minimum_reserves()
            self.banks[i].assert_alm()
            self.banks[i].assert_lcr()
            # self.banks[i].assert_leverage()
            self.banks[i].steps += 1

    def simulate(self, time_steps, save_every=10, jaccard_period=10):
        self.period = jaccard_period
        for _ in tqdm(range(time_steps)):
            if self.total_steps % save_every == 0.0:
                self.save_figs()
                self.save_time_series()
            self.step_network()
            self.update_metrics()
            self.total_steps += 1
        for bank in self.banks:
            print(bank)
            print(
                "Bank Deposits {} Bank Cash {}".format(
                    bank.liabilities["Deposits"], bank.assets["Cash"]
                )
            )
        self.save_figs()
        self.save_time_series()
        self.final_metrics()
