import numpy as np
from scipy.stats import pareto
from bank import BankAgent
from tqdm import tqdm
from scipy.stats import truncnorm
import networkx as nx
import os
import shutil
import graphics as gx

float_limit = 1e-8


class InterBankNetwork:
    def __init__(
        self,
        n_banks,
        alpha_pareto=0.5,
        beta_lcr=10.0,
        beta_star_lcr=10.0,
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
        assert shock_method in ["log-normal", "dirichlet", "uniform"], (
            "Not valid initialisation method :"
            " 'log-normal' or 'dirichlet' or 'uniform'"
        )
        # Params
        self.n_banks = n_banks
        self.alpha_pareto = alpha_pareto
        self.beta_lcr = beta_lcr
        self.beta_star_lcr = beta_star_lcr
        self.initial_mr = initial_mr
        self.initial_l2s = initial_l2s
        self.collateral_value = collateral_value
        self.init = init
        self.shock_method = shock_method
        if shock_method == "dirichlet":
            self.std_control = 1.0 / (std_law ** 2.0)
            self.conservative_shock = True
        elif shock_method == "uniform":
            self.conservative_shock = True
        elif shock_method == "log-normal":
            self.std_control = np.sqrt(np.log(1.0 + std_law ** 2.0))
            self.conservative_shock = False
        self.result_location = result_location

        # Internal
        self.banks = []
        self.deposits = np.zeros(n_banks)
        self.balance_sheets = np.zeros(n_banks)
        self.total_assets = {}
        self.total_liabilities = {}
        self.collateral = 1.0
        self.adj_matrix = np.zeros((n_banks, n_banks))
        self.prev_adj_matrix = np.zeros((n_banks, n_banks))
        self.metrics = {}
        self.total_steps = 0.0
        self.reset_network()

    def reset_network(self):
        for b in range(1, self.n_banks + 1):
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
            self.banks.append(
                BankAgent(
                    bank_id=b,
                    initial_deposits=deposits,
                    initial_mr=self.initial_mr,
                    beta_lcr=self.beta_lcr,
                    beta_star_lcr=self.beta_star_lcr,
                    initial_l2s=self.initial_l2s,
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
        self.total_steps = 0.0
        if os.path.exists(self.result_location):
            shutil.rmtree(self.result_location)
        os.makedirs(os.path.join(self.result_location, "Networks"))
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
            self.adj_matrix[i, :] = np.array(
                list(self.banks[i].reverse_repos.values())
            )
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

    def save_figs(self):
        gx.plot_network(
            self.adj_matrix,
            os.path.join(self.result_location, "Networks"),
            self.total_steps,
        )

        gx.bar_plot_balance_sheet(
            self.balance_sheets,
            self.total_assets,
            self.total_liabilities,
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

    def step_network(self):
        if self.shock_method == "uniform":
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
            rho_neg = np.random.uniform(-1, 0, size=N_half)

            # apply a
            rho_pos = (
                -rho_neg * deposits_p[0:N_half] / deposits_p[N_half:N_max]
            )

            # concatenate the relative shocks
            if self.n_banks > N_max:
                rho = np.concatenate([rho_neg, rho_pos, 0])
            elif self.n_banks == N_max:
                rho = np.concatenate([rho_neg, rho_pos])

            # build an un-permuted array of absolute shocks
            shocks = np.zeros(self.n_banks)

            # compute the absolute shock from the deposit amount
            shocks[ix_p] = deposits_p * rho

            assert abs(shocks.sum()) < float_limit, print(
                abs(shocks.sum()), "Shock doesn't sum to zero"
            )
            assert (
                self.deposits + shocks
            ).all() > -float_limit, "negative shocks larger than deposits"

        if self.shock_method == "dirichlet":
            # dirichlet approach
            deposits = self.deposits + float_limit
            dispatch = np.random.dirichlet(
                (deposits / deposits.sum()) * self.std_control
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
            shocks = (
                truncnorm.rvs(-3, 3, size=len(self.banks)) * self.deposits / 3
            )

        ix = np.arange(self.n_banks)
        for i in ix:
            self.banks[i].set_shock(shocks[i])
            self.banks[i].set_collateral(self.collateral)
            self.banks[i].lcr_step()
        ix = np.random.permutation(ix)
        for i in ix:
            self.banks[i].step_end_repos_chain()
        ix = np.random.permutation(ix)
        for i in ix:
            self.banks[i].step_repos()
        for i in ix:
            self.banks[i].assert_minimal_reserve()
            self.banks[i].assert_alm()
            self.banks[i].assert_lcr()
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
