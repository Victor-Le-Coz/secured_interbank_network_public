import numpy as np
import random
from scipy.stats import pareto
from bank import BankAgent
from tqdm import tqdm


class InterBankNetwork:
    def __init__(
        self,
        n_banks,
        alpha_pareto=0.5,
        perc_deposit_shock=1.0,
        beta_lcr=10.0,
        beta_star_lcr=10.0,
        initial_mr=1.0,
        initial_l2s=3.0,
        collateral_value=1.0,
        init="constant",
    ):
        self.banks = []
        self.deposits = np.zeros(n_banks)
        for b in range(1, n_banks + 1):
            if init == "pareto":
                deposits = (
                    pareto.rvs(
                        alpha_pareto, loc=0, scale=1, size=1, random_state=None
                    )[0]
                    * 100.0
                )
            elif init == "constant":
                deposits = 100.0
            self.deposits[b - 1] = deposits
            self.banks.append(
                BankAgent(
                    bank_id=b,
                    initial_deposits=deposits,
                    initial_mr=initial_mr,
                    beta_lcr=beta_lcr,
                    beta_star_lcr=beta_star_lcr,
                    initial_l2s=initial_l2s,
                    collateral_value=collateral_value,
                )
            )
        for bank in self.banks:
            bank.initialize_banks(self.banks)
        self.collateral = 1.0
        self.adj_matrix = np.zeros((n_banks, n_banks))
        self.sigma_shock = perc_deposit_shock / 3.0

    def step_network(self):
        dispatch = np.random.dirichlet(self.deposits * 100.0)
        deposits = dispatch * self.deposits.sum()
        shocks = deposits - self.deposits
        # print("Minimum shock is :", min(self.deposits + shocks))
        # print("Sum of shocks is {}".format(round(shocks.sum(), 2)))
        # print("Shocks : ", shocks)
        assert abs(shocks.sum()) < 1e-8, "Shock doesn't sum to zero"

        ix = np.arange(len(self.banks))
        ix = np.random.permutation(ix)
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
        excess_liquidity = 0.0
        total_deposits = 0.0
        total_securities = 0.0
        total_cash = 0.0
        total_loans = 0.0
        total_mro = 0.0
        for i in ix:
            self.banks[i].assert_minimal_reserve()
            self.banks[i].assert_alm()
            self.banks[i].assert_lcr()
            self.banks[i].steps += 1
            self.adj_matrix[i, :] = np.array(
                list(self.banks[i].reverse_repos.values())
            )
            self.deposits[i] = self.banks[i].liabilities["Deposits"]
            excess_liquidity += (
                self.banks[i].assets["Cash"]
                - self.banks[i].alpha * self.banks[i].liabilities["Deposits"]
            )
            total_deposits += self.banks[i].liabilities["Deposits"]
            total_securities += (
                self.banks[i].assets["Securities Usable"]
                + self.banks[i].off_balance["Securities Collateral"]
            )
            total_cash += self.banks[i].assets["Cash"]
            total_mro += self.banks[i].liabilities["MROs"]
            total_loans += self.banks[i].assets["Loans"]
        # print("Total Loans - Total MROs is {}".format(total_loans - total_mro))
        # print("Excess Liquidity is {}".format(excess_liquidity))
        # print("Total Deposits is {}".format(total_deposits))
        # print("Total Securities is {}".format(total_securities))
        # print("Total Cash is {}".format(total_cash))
        # print(self.adj_matrix)

    def simulate(self, time_steps):
        for _ in tqdm(range(time_steps)):
            self.step_network()
        for bank in self.banks:
            print(bank)
            print(
                "Bank Deposits {} Bank Cash {}".format(
                    bank.liabilities["Deposits"], bank.assets["Cash"]
                )
            )
