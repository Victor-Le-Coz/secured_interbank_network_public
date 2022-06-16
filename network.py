import numpy as np
import random
from scipy.stats import pareto
from bank import BankAgent


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
    ):
        self.banks = []
        self.deposits = np.zeros(n_banks)
        for b in range(1, n_banks + 1):
            deposits = pareto.rvs(
                alpha_pareto, loc=0, scale=1, size=1, random_state=None
            )[0]
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
        dispatch = np.random.dirichlet(self.deposits / self.deposits.sum())
        deposits = 0.9 * self.deposits + (0.1 * self.deposits).sum() * dispatch
        shocks = deposits - self.deposits
        # print("Minimum shock is :", min(self.deposits + shocks))
        # print("Sum of shocks is {}".format(round(shocks.sum(), 2)))
        assert abs(shocks.sum()) < 1e-8, "Shock doesn't sum to zero"

        for i, bank in enumerate(self.banks):
            bank.set_shock(shocks[i])
            bank.set_collateral(self.collateral)
            bank.lcr_step()
        random.shuffle(self.banks)
        for i, bank in enumerate(self.banks):
            bank.step_end_repos_chain()
        random.shuffle(self.banks)
        for i, bank in enumerate(self.banks):
            bank.step_repos()
        excess_liquidity = 0.0
        total_deposits = 0.0
        total_securities = 0.0
        total_cash = 0.0
        total_loans = 0.0
        total_mro = 0.0
        for i, bank in enumerate(self.banks):
            bank.assert_regulatory()
            bank.steps += 1
            self.adj_matrix[i, :] = np.array(list(bank.reverse_repos.values()))
            self.deposits[i] = bank.liabilities["Deposits"]
            excess_liquidity += (
                bank.assets["Cash"] - bank.alpha * bank.liabilities["Deposits"]
            )
            total_deposits += bank.liabilities["Deposits"]
            total_securities += (
                bank.assets["Securities Usable"]
                + bank.off_balance["Securities Collateral"]
            )
            total_cash += bank.assets["Cash"]
            total_mro += bank.liabilities["MROs"]
            total_loans += bank.assets["Loans"]
        print("Total Loans {} Total MROs is {}".format(total_loans, total_mro))
        # print("Excess Liquidity is {}".format(excess_liquidity))
        # print("Total Deposits is {}".format(total_deposits))
        # print("Total Securities is {}".format(total_securities))
        # print("Total Cash is {}".format(total_cash))
