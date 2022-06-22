import numpy as np


class BankAgent:
    def __init__(
        self,
        bank_id,
        initial_deposits,
        beta_lcr=10.0,
        beta_star_lcr=10.0,
        initial_mr=1.0,
        initial_l2s=3.0,
        collateral_value=1.0,
        conservative_shock=True,
    ):

        self.id = str(bank_id)
        self.alpha = initial_mr / 100.0
        self.beta = beta_lcr / 100.0
        self.gamma = initial_l2s / 100.0
        self.collateral = collateral_value
        self.previous_shock = 0.0
        self.beta_star = beta_star_lcr / 100.0
        self.reverse_accept = 0.0
        self.shock = 0.0
        self.conservative_shock = conservative_shock

        # Learning algorithm
        self.trust = {}
        self.visits = {}
        self.steps = 1
        self.exploration_coeff = np.sqrt(2.0)

        self.on_balance_repos = {}
        self.off_balance_repos = {}
        self.reverse_repos = {}
        self.banks = {}

        self.assets = {
            "Cash": 0.0,
            "Securities Usable": 0.0,
            "Securities Encumbered": 0.0,
            "Loans": 0.0,
            "Reverse Repos": 0.0,
        }

        self.liabilities = {
            "Own Funds": 0.0,
            "Deposits": initial_deposits,
            "Repos": 0.0,
            "MROs": 0.0,
        }

        self.off_balance = {
            "Securities Collateral": 0.0,
            "Securities Reused": 0.0,
        }

        self.initialize_balance_sheet()
        self.assert_leverage()
        self.assert_lcr()
        self.assert_alm()
        self.assert_minimal_reserve()

    def initialize_balance_sheet(self):
        self.assets["Cash"] = self.liabilities["Deposits"] * self.alpha
        self.assets["Securities Usable"] = (
            self.beta - self.alpha
        ) * self.liabilities["Deposits"]
        self.liabilities["Own Funds"] = (
            self.gamma / (1.0 - self.gamma)
        ) * self.liabilities["Deposits"]
        self.assets["Loans"] = (
            self.liabilities["Own Funds"]
            + self.liabilities["Deposits"]
            - self.assets["Cash"]
            - self.assets["Securities Usable"]
        )

    def initialize_banks(self, banks):
        for bank in banks:
            self.reverse_repos[bank.id] = 0.0
            if bank != self.id:
                self.trust[bank.id] = 0.0
                self.visits[bank.id] = 1.0
                self.on_balance_repos[bank.id] = 0.0
                self.off_balance_repos[bank.id] = 0.0
                self.banks[bank.id] = bank

    def total_assets(self):
        return sum(self.assets.values())

    def total_liabilities(self):
        return sum(self.liabilities.values())

    def __str__(self):
        p_str = """Bank {}\r
        ----------------------------------------\r
        Total Assets: {}\r
        ----Cash: {}\r
        ----Securities Usable: {}\r
        ----Securities Encumbered: {}\r
        ----Reverse Repos: {}\r
        ----Loans: {}\r
        ----------------------------------------\r
        Total Liabilities: {}\r
        ----Own Funds: {}\r
        ----Deposits: {}\r
        ----Repos: {}\r
        ----MROs: {}\r
        ----------------------------------------\r
        Off-balance sheet Items:\r
        ----Collateral received: {}\r
        ----Collateral Reuse: {}\r
        ----------------------------------------\r
        Regulatory:\r
        ----Liquidity Coverage Ratio: {}%\r
        ----Cash / Deposits (for minimal reserves): {}%\r
        ----Leverage / Solvency ratio: {}%\r
        """
        p_str = p_str.format(
            self.id,
            round(self.total_assets(), 2),
            round(self.assets["Cash"], 2),
            round(
                self.assets["Securities Usable"] * self.collateral, 2
            ),
            round(
                self.assets["Securities Encumbered"] * self.collateral,
                2,
            ),
            round(self.assets["Reverse Repos"], 2),
            round(self.assets["Loans"], 2),
            round(self.total_liabilities(), 2),
            round(self.liabilities["Own Funds"], 2),
            round(self.liabilities["Deposits"], 2),
            round(self.liabilities["Repos"], 2),
            round(self.liabilities["MROs"], 2),
            round(
                self.off_balance["Securities Collateral"]
                * self.collateral,
                2,
            ),
            round(
                self.off_balance["Securities Reused"]
                * self.collateral,
                2,
            ),
            round(self.liquidity_coverage_ratio() * 100, 2),
            round(self.cash_to_deposits() * 100, 2),
            round(self.leverage_to_solvency_ratio() * 100, 2),
        )
        return p_str

    def liquidity_coverage_ratio(self):
        lcr = (
            self.assets["Cash"]
            + self.assets["Securities Usable"] * self.collateral
        )
        lcr += (
            self.off_balance["Securities Collateral"] * self.collateral
        )
        lcr /= self.beta * self.liabilities["Deposits"]
        return lcr

    def cash_to_deposits(self):
        return self.assets["Cash"] / self.liabilities["Deposits"]

    def leverage_to_solvency_ratio(self):  # Check it after each step
        ltsr = (
            self.assets["Cash"]
            + self.assets["Securities Usable"] * self.collateral
            + self.assets["Securities Encumbered"] * self.collateral
            + self.assets["Loans"]
            + self.assets["Reverse Repos"]
            + (
                self.assets["Reverse Repos"]
                - self.off_balance["Securities Collateral"]
                * self.collateral
                - self.off_balance["Securities Reused"]
                * self.collateral
            )
        )
        ltsr = self.liabilities["Own Funds"] / ltsr
        return ltsr

    def assert_lcr(self):
        assert (
            self.assets["Cash"]
            + self.off_balance["Securities Collateral"]
            * self.collateral
            + self.assets["Securities Usable"] * self.collateral
            + 1e-8
            >= self.liabilities["Deposits"] * self.beta
        ), self.__str__() + "\nLCR not at its target value for bank {} at step {}".format(
            self.id, self.steps
        )

    def assert_minimal_reserve(self):
        assert (
            self.assets["Cash"] + 1e-8
            >= self.alpha * self.liabilities["Deposits"]
        ), self.__str__() + "\nMinimum reserves not respected for bank {} at step {}".format(
            self.id, self.steps
        )

    def assert_leverage(self):
        assert (
            self.leverage_to_solvency_ratio() + 1e-8 > self.gamma
        ), self.__str__() + "\nLeverage to solvency ratio not at its target value for bank {} at step {}" "".format(
            self.id, self.steps
        )

    def assert_alm(self):
        assert (
            np.abs(
                self.total_assets() / self.total_liabilities() - 1.0
            )
            < 1e-8
        ), self.__str__() + "\nAssets don't match Liabilities for bank {} at step {}".format(
            self.id, self.steps
        )

    def set_shock(self, shock):
        self.shock = shock
        self.liabilities["Deposits"] += self.shock
        self.assets["Cash"] += self.shock

    def set_collateral(self, collateral):
        self.collateral = collateral

    def lcr_step(self):
        if self.shock < 0.0:
            self.negative_lcr_management()
        else:
            self.positive_lcr_management()

    def positive_lcr_management(self):
        # print("Positive LCR Management Bank {}".format(self.id))
        cash_target = -(
            self.beta_star * self.liabilities["Deposits"]
            - self.assets["Cash"]
            - self.assets["Securities Usable"] * self.collateral
            - self.off_balance["Securities Collateral"]
            * self.collateral
        )
        # print("Cash Target is {}".format(cash_target))
        self.assets["Cash"] -= cash_target
        bce_reimburse = min(cash_target, self.liabilities["MROs"])
        self.liabilities["MROs"] -= bce_reimburse
        self.assets["Loans"] += cash_target - bce_reimburse

    def negative_lcr_management(self):
        # print("Negative LCR Management Bank {}".format(self.id))
        cash_target = (
            self.beta_star * self.liabilities["Deposits"]
            - self.assets["Cash"]
            - self.assets["Securities Usable"] * self.collateral
            - self.off_balance["Securities Collateral"]
            * self.collateral
        )
        # print("Cash Target is {}".format(cash_target))
        self.liabilities["MROs"] += cash_target
        self.assets["Cash"] += cash_target

    def step_end_repos_chain(self):
        self.end_repos(
            self.assets["Cash"]
            - self.alpha * self.liabilities["Deposits"]
        )

    def end_repos(self, target):
        if target <= 0.0:
            return
        # First end off-balance repos
        trust = self.trust.copy()
        for b, t in sorted(trust.items(), key=lambda item: item[1]):
            end = min(self.off_balance_repos[b], target)
            if end > 0.0:
                self.banks[b].end_repo(self.id, end)
                self.assets["Cash"] -= end
                self.liabilities["Repos"] -= end
                self.off_balance["Securities Collateral"] += end
                self.off_balance["Securities Reused"] -= end
                self.off_balance_repos[b] -= end
                target = max(target - self.off_balance_repos[b], 0.0)
            if target == 0.0:
                break
        if target == 0.0:
            return
        # Second end on-balance repos
        for b, t in sorted(trust.items(), key=lambda item: item[1]):
            end = min(self.on_balance_repos[b], target)
            if end > 0.0:
                self.banks[b].end_repo(self.id, end)
                self.assets["Cash"] -= end
                self.liabilities["Repos"] -= end
                self.assets["Securities Usable"] += end
                self.assets["Securities Encumbered"] -= end
                self.on_balance_repos[b] -= end
                target = max(target - self.on_balance_repos[b], 0.0)
            if target == 0.0:
                break

    def step_repos(self):
        self.enter_repos(
            -(
                self.assets["Cash"]
                - self.alpha * self.liabilities["Deposits"]
            )
        )

    def enter_repos(self, repo_ask):
        if repo_ask <= 0.0:
            return
        bank_list = list(self.trust.keys())
        bank_list.remove(self.id)
        while True:
            b = self.choose_bank(bank_list)
            bank_list.remove(b)
            rest = self.banks[b].ask_for_repo(self.id, repo_ask)
            self.update_learning(b, (repo_ask - rest) / repo_ask)
            assert (
                self.assets["Securities Usable"]
                + self.off_balance["Securities Collateral"]
                + 1e-8
                > repo_ask - rest
            ), self.__str__() + "\nNot Enough Collateral for bank {}".format(
                self.id
            )
            coll_usable = min(
                repo_ask - rest, self.assets["Securities Usable"]
            )
            coll_coll = max(
                repo_ask - rest - self.assets["Securities Usable"], 0.0
            )
            self.assets["Cash"] += repo_ask - rest
            self.on_balance_repos[b] += coll_usable
            self.assets["Securities Usable"] -= coll_usable
            self.assets["Securities Encumbered"] += coll_usable
            self.off_balance_repos[b] += coll_coll
            self.off_balance["Securities Collateral"] -= coll_coll
            self.off_balance["Securities Reused"] += coll_coll
            self.liabilities["Repos"] += repo_ask - rest
            repo_ask = rest
            if rest == 0.0 or len(bank_list) == 0:
                break
        if not self.conservative_shock:
            self.liabilities["MROs"] += repo_ask
            self.assets["Cash"] += repo_ask
            repo_ask = 0.0
        assert (
            repo_ask < 1e-8
        ), self.__str__() + "\nRepo needs not filled for bank {}".format(
            self.id
        )
        for bank in bank_list:
            self.visits[bank] = (
                self.visits[bank] - 0.1 * self.visits[bank]
            )

    def choose_bank(self, bank_list):
        ucts = {}
        for b in self.trust.keys():
            if b in bank_list:
                # ucts[b] = self.trust[
                #     b
                # ] / self.steps + self.exploration_coeff * np.sqrt(
                #     self.steps / self.visits[b]
                # )
                ucts[b] = self.trust[
                    b
                ] + self.exploration_coeff * np.sqrt(
                    1.0 / self.visits[b]
                )
            else:
                ucts[b] = 0.0
        return max(ucts, key=ucts.get)

    def update_learning(self, bank, value):
        # self.visits[bank] += 1
        # self.trust[bank] += value
        self.visits[bank] = self.visits[bank] + 0.1 * (
            1.0 - self.visits[bank]
        )
        self.trust[bank] = self.trust[bank] + 0.1 * (
            value - self.trust[bank]
        )

    def ask_for_repo(self, bank_id, amount):
        reverse_accept = (
            self.assets["Cash"]
            - self.alpha * self.liabilities["Deposits"]
        )
        assert (
            self.id != bank_id
        ), "Bank {} is lending to itself".format(self.id)
        # print(
        #     "Repo ask {}:{} => {}:{}".format(
        #         bank_id, amount, self.id, reverse_accept
        #     )
        # )
        if reverse_accept <= 0.0:
            return amount
        repo = min(amount, reverse_accept)
        self.off_balance["Securities Collateral"] += repo
        self.assets["Cash"] -= repo
        self.assets["Reverse Repos"] += repo
        self.reverse_repos[bank_id] += repo
        return max(amount - reverse_accept, 0.0)

    def end_repo(self, bank_id, amount):
        if amount <= 0.0:
            return
        # print(
        #     "End Repo ask {}:{} => {}:SC:{}:SU:{}".format(
        #         bank_id,
        #         round(amount, 2),
        #         self.id,
        #         round(self.off_balance["Securities Collateral"], 2),
        #         round(self.assets["Securities Usable"], 2),
        #     )
        # )
        collateral_rest = max(
            amount
            - self.off_balance["Securities Collateral"]
            - self.assets["Securities Usable"],
            0.0,
        )
        self.end_repos(collateral_rest)
        assert (
            self.off_balance["Securities Collateral"]
            + self.assets["Securities Usable"]
            > amount - 1e-8
        ), self.__str__() + "\nError, not enough collateral asked back for bank {}".format(
            bank_id
        )
        collateral_coll = min(
            amount, self.off_balance["Securities Collateral"]
        )
        collateral_rest = max(
            amount - self.off_balance["Securities Collateral"], 0.0
        )
        self.assets["Cash"] += amount
        self.assets["Reverse Repos"] -= amount
        self.reverse_repos[bank_id] -= amount
        self.off_balance["Securities Collateral"] -= collateral_coll
        self.assets["Securities Usable"] -= collateral_rest
        self.assets["Securities Encumbered"] += collateral_rest
        self.off_balance["Securities Reused"] -= collateral_rest
