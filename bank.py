import numpy as np

# The parameter sets the limit to the float precision when running the algorithm, a value lower than this amount is
# considered as negligible.
float_limit = 1e-4


class ClassBank:
    """
    This class models a bank agent.
    """

    # <editor-fold desc="Instance methods for the initialization of an instance of Class Banks">
    def __init__(self,
                 id,
                 initial_deposits,
                 alpha=1.0,
                 beta_init=10.0,
                 beta_reg=10.0,
                 beta_star=10.0,
                 gamma=3.0,
                 collateral_value=1.0,
                 conservative_shock=True):
        """
        Instance methode initializing the class BankAgent.
        :param id: the identifier of the instance of the bank agent.
        :param initial_deposits: the initial deposits used to initialize the instance of the bank agent.
        :param beta_init: the initial LCR share of deposits used to define the amount of securities usable.
        :param beta_reg: the regulatory LCR share of deposits required (currently 10% in the Eurozone).
        :param beta_star: the targeted LCR  share of deposits => could be modelled, currently set as a constant.
        :param alpha: the share of deposits required as minimum reserves (currently 1% in the Eurozone).
        :param gamma: the share of total asset required as minimum leverage ratio (currently 3% in the Eurozone).
        :param collateral_value: value of the collateral in monetary units => could be modelled, currently set as constant.
        :param conservative_shock: the behavioural rule is modified when shocks are not conservative,
        """

        # initialisation of the class parameters
        self.id = str(id)
        self.alpha = alpha / 100.0
        self.beta_init = beta_init / 100.0
        self.beta_reg = beta_reg / 100.0
        self.beta_star = beta_star / 100.0
        self.gamma = gamma / 100.0
        self.collateral_value = collateral_value
        self.conservative_shock = conservative_shock

        # Other parameters initialization
        self.excess_liquidity = 0.0  # Excess liquidity a bank can provide in a reverse repo.
        self.shock = 0.0  # Value of the shock received by a bank at a given time_step.

        # Parameters related to the learning algorithm for the selection of counterparties on the repo market
        self.trust = {}  # Dictionary of the trust coefficients with the other banks.
        self.visits = {}  # Dictionary of the number of time the bank agent obtained a repo with each of the banks.
        self.steps = 1  # Step number in the simulation process, used as a proxy in the bandit algo for past nb of
        # repos.
        self.exploration_coeff = np.sqrt(2.0)  # Exploration coefficient setting the ability to explore banks that have.
        # not been tested in the past.

        # Dictionaries & lists related to the other bank agents
        self.on_balance_repos = {}  # Dictionary of the repos received from each of the other banks, using securities.
        self.off_balance_repos = {}  # Dictionary of the repos received from each of the other banks, re-using
        # collateral.
        self.reverse_repos = {}  # Dictionary of the reverse repos provided to each of the other banks.
        self.banks = {}  # Dictionary of the instances of the ClassBank class existing in the ClassNetwork class.

        # Dictionaries & list related to the computation of the maturity of the repos
        self.repos_on_filo = {}  # Dictionary over the other banks of a list of list couples with the amount of
        # securities usable, and time steps at which repo are entered in.
        self.repos_off_filo = {}  # Dictionary over the other banks of a list of list couples with the amount of
        # securities collateral, and time steps at which repo are entered in.
        self.repos_on_maturities = []  # List of the maturities of the repos performed by the instance of ClassBank,
        # using its securities usable.
        self.repos_on_amounts = []
        self.repos_off_maturities = [] # List of the maturities of the repos performed by the instance of ClassBank,
        # using its securities collateral.
        self.repos_off_amounts = []

        # Dictionaries of the balance sheet items of an instance of the ClassBank.
        self.assets = {  # Assets.
            "Cash": 0.0,
            "Securities Usable": 0.0,
            "Securities Encumbered": 0.0,
            "Loans": 0.0,
            "Reverse Repos": 0.0,
        }
        self.liabilities = {  # Liabilities.
            "Own Funds": 0.0,
            "Deposits": initial_deposits,
            "Repos": 0.0,
            "MROs": 0.0,
        }
        self.off_balance = {  # Off-balance sheet items.
            "Securities Collateral": 0.0,
            "Securities Reused": 0.0,
        }

        # Run each of the
        self.initialize_balance_sheet()

    def initialize_balance_sheet(self):
        """
        Instance method initializing the balance sheet items of an instance of ClassBank.
        :return:
        """

        # The cash is set to its minimum reserve amount.
        self.assets["Cash"] = self.liabilities["Deposits"] * self.alpha

        # The collateral is set to the amount allowing to match the beta_init.
        self.assets["Securities Usable"] = (
                (self.beta_init - self.alpha)
                * self.liabilities["Deposits"]
                / self.collateral_value
        )

        # The Own-funds are set to match the leverage ratio.
        self.liabilities["Own Funds"] = (
                                                self.gamma / (1.0 - self.gamma)
                                        ) * self.liabilities["Deposits"]

        # The loans are set to match the assets and liabilities.
        self.assets["Loans"] = (
                self.liabilities["Own Funds"]
                + self.liabilities["Deposits"]
                - self.assets["Cash"]
                - self.assets["Securities Usable"] * self.collateral_value
        )

    def initialize_banks(self, banks):
        """
        Initialize the list of dictionaries of the bank agent instance self. with the information about each of the
        other banks in the network.
        :param banks : a list of ClassBank class instance, it is a parameter of the ClassNetwork class.
        :return:
        """
        for bank in banks:
            self.reverse_repos[bank.id] = 0.0
            self.repos_on_filo[bank.id] = []
            self.repos_off_filo[bank.id] = []
            if bank.id != self.id:
                self.trust[bank.id] = 0.0
                self.visits[bank.id] = 1.0
                self.on_balance_repos[bank.id] = 0.0
                self.off_balance_repos[bank.id] = 0.0
                self.banks[bank.id] = bank

    # </editor-fold>

    def set_shock(self, shock):
        """
        Instance method applying a shock to an instance of ClassBank.
        It updates the deposits and the cash.
        :param shock: shock applied
        :return:
        """
        self.shock = shock
        self.liabilities["Deposits"] += self.shock
        self.assets["Cash"] += self.shock

    def set_collateral(self, collateral_value):
        """
        Instance method updating the value of the collateral of an instance of ClassBank.
        :param shock: shock applied
        :return:
        """
        self.collateral_value = collateral_value

    def step_lcr_mgt(self):
        """
        Instance method updating the cash, loans, and MROs of an instance of ClassBank.
        :return:
        """
        # Definition of the variation of cash required to match the targeted LCR, this value can be positive or
        # negative.
        delta_cash = (
                self.beta_star * self.liabilities["Deposits"]
                - self.assets["Cash"]
                - self.assets["Securities Usable"] * self.collateral_value
                - self.off_balance["Securities Collateral"] * self.collateral_value
        )

        # Fill-in the Cash, Loans, and Main Refinancing Operations (MROs). In case of a negative delta cash,
        # the bank first reimburses its existing central bank funding (MRO) before granting new loans.
        self.assets["Cash"] += delta_cash
        self.assets["Loans"] += -min(
            self.liabilities["MROs"] + delta_cash, 0.0
        )
        self.liabilities["MROs"] = max(
            self.liabilities["MROs"] + delta_cash, 0.0
        )

    def step_end_repos(self):
        """
        Instance method to perform the step of ending repos. It actually calls the end_repos method which can be used
        either to end repo due to excess liquidity or due to collateral needs.
        :return:
        """
        excess_liquidity = (
                self.assets["Cash"] - self.alpha * self.liabilities["Deposits"]
        )
        excess_liquidity = max(excess_liquidity - float_limit, 0.0)
        self.end_repos(excess_liquidity)

    def end_repos(self, repo_amount):
        """
        Instance method allows to close all the repos from an instance of ClassBank, as well as all the necessary
        repos of the lenders of cash to this bank agent, in a recursive manner.
        :param: repo_amount: amount of excess_liquidity or collateral to call back in case of chain of repos ending.
        :return:
        """

        # stop the method if the mount of repo to close is negative
        if repo_amount <= 0.0:
            return

        # First step, end all off-balance repos, starting by the ones with banks having the lowest trust factor.
        trust = self.trust.copy()

        # For loop over the sorted list of the other banks, starting by the lowest trust factor.
        for b, t in sorted(
                trust.items(), reverse=True, key=lambda item: item[1]
        ):
            # Definition of the target amount of off-balance repos to close with a given bank b
            end = min(self.off_balance_repos[b], repo_amount)

            # Case disjunction
            if end > 0.0:

                # The bank id b must close its reverse repo with the bank agent self, the instance method
                # end_reverse_repo calls recursively the end_repos method for the bank b in case this latter would
                # not own sufficient colletral to end its reverse repo.
                self.banks[b].end_reverse_repo(self.id, end)

                # Algorithm to update the list of list of the maturities of the repos performed by the instance of
                # ClassBank, using its securities collateral.
                end_amount = end
                new = self.repos_off_filo[b].copy()
                for (amount, t) in self.repos_off_filo[b]:
                    if amount <= end_amount:
                        del new[0]
                        end_amount -= amount
                        self.repos_off_maturities.append(self.steps - t)
                        self.repos_off_amounts.append(amount)
                    else:
                        new[0][0] -= amount
                        break
                self.repos_off_filo[b] = new

                # Update all the balance sheet items related to the closure of the repo
                self.assets["Cash"] -= end
                self.liabilities["Repos"] -= end
                self.off_balance["Securities Collateral"] += (
                        end / self.collateral_value
                )
                self.off_balance["Securities Reused"] -= end / self.collateral_value
                self.off_balance_repos[b] -= end
                repo_amount = max(
                    repo_amount - self.off_balance_repos[b] - float_limit, 0.0
                )

            # Break the for loop if off balance repos are sufficient to empty the excess liquidity
            if repo_amount == 0.0:
                break

        # End the instance method if off balance repos are sufficient to empty the excess liquidity
        if repo_amount == 0.0:
            return

        # Second step, end all on-balance repos, starting by the ones with banks having the lowest trust factor.
        for b, t in sorted(
                trust.items(), reverse=True, key=lambda item: item[1]
        ):
            end = min(self.on_balance_repos[b], repo_amount)
            if end > 0.0:

                # The bank id b must close its reverse repo with the bank agent self, the instance method
                # end_reverse_repo calls recursively the end_repos method for the bank b in case this latter would
                # not own sufficient colletral to end its reverse repo.
                self.banks[b].end_reverse_repo(self.id, end)

                # Algorithm to update the list of list of the maturities of the repos performed by the instance of
                # ClassBank, using its securities usable.
                end_amount = end
                new = self.repos_on_filo[b].copy()
                for amount, t in self.repos_on_filo[b]:
                    if amount <= end_amount:
                        del new[0]
                        end_amount -= amount
                        self.repos_on_maturities.append(self.steps - t)
                        self.repos_on_amounts.append(amount)
                    else:
                        new[0][0] -= amount
                        break
                self.repos_on_filo[b] = new

                # Update all the balance sheet items related to the closure of the repo
                self.assets["Cash"] -= end
                self.liabilities["Repos"] -= end
                self.assets["Securities Usable"] += end / self.collateral_value
                self.assets["Securities Encumbered"] -= end / self.collateral_value
                self.on_balance_repos[b] -= end
                repo_amount = max(
                    repo_amount - self.on_balance_repos[b] - float_limit, 0.0
                )

            # Break the for loop if on balance repos are sufficient to empty the excess liquidity
            if repo_amount == 0.0:
                break

    def end_reverse_repo(self, bank_id, amount):
        """
        Instance method allowing a instance of the ClassBank self, to close its reverse repo with the bank bank_id,
        upon request of of this latter. This instance method calls recursively the end_repos method with the missing
        amount of collateral to be provided back.
        :param bank_id: identifier of of the bank requesting to close its
        repo for a given amount.
        :param amount: amount requested to be closed.
        :return:
        """

        if amount <= 0.0:
            return

        # Debugging prints
        # print(
        #     "End Repo ask {}:{} => {}:SC:{}:SU:{}".format(
        #         bank_id,
        #         round(amount, 2),
        #         self.id,
        #         round(self.off_balance["Securities Collateral"], 2),
        #         round(self.assets["Securities Usable"], 2),
        #     )
        # )

        # Definition of the amount of collateral missing to allow the closing of the reverse repo. We allow the bank
        # self to use its securities usable as the substitute for the securities he received in the first place.
        missing_collateral = max(
            amount
            - self.off_balance["Securities Collateral"] * self.collateral_value
            - self.assets["Securities Usable"] * self.collateral_value
            - 2.0 * float_limit,
            0.0,
        )

        # Recursive calling of the missing amount of collateral through the ending of the own repos of the bank agent.
        self.end_repos(missing_collateral)

        # Recompute the missing amount of collateral after the end of the recursive algorithm.
        missing_collateral = max(
            amount
            - self.off_balance["Securities Collateral"] * self.collateral_value
            - self.assets["Securities Usable"] * self.collateral_value
            - 2.0 * float_limit,
            0.0,
        )

        # Assert if the recursive algorithm worked adequately, otherwise print an error
        assert (
                missing_collateral >= 0.0
        ), self.__str__() + "\nError, the repo could not be ended as the bank {} has not enough collateral to end its " \
                            "reverse repo with bank {}".format(
            self.id,
            bank_id,
        )

        # Update all the required balance sheet items by the closing of the reverse repo. The bank self prefers to
        # use first its securities collateral to end its reverse repo, then it uses its securities usable

        # Definition of the decrease in the securities collateral and securities usable
        securities_collateral_decrease = min(
            amount, self.off_balance["Securities Collateral"] * self.collateral_value
        )
        securities_usable_decerease = max(
            amount
            - self.off_balance["Securities Collateral"] * self.collateral_value,
            0.0,
        )

        # Update of the balance sheet items
        self.assets["Cash"] += amount
        self.assets["Reverse Repos"] -= amount
        self.reverse_repos[bank_id] -= amount
        self.off_balance["Securities Collateral"] -= (
                securities_collateral_decrease / self.collateral_value
        )
        self.assets["Securities Usable"] -= securities_usable_decerease / self.collateral_value

        # Attention point: when the bank self uses its securities usable, the securities reused decrease by the same
        # amount and the securities encumbered increase by the same amount, to ensure each security has an official
        # owner.
        self.off_balance["Securities Reused"] -= (
                securities_usable_decerease / self.collateral_value
        )
        self.assets["Securities Encumbered"] += (
                securities_usable_decerease / self.collateral_value
        )

    def step_enter_repos(self):
        """
        Instance method allowing an instance of ClassBank to enter into a series of repo to meet is liquidity needs.
        :return:
        """

        # Define the amount of repo to be requested (it is a positive amount if there is a liquidity need)
        repo_ask = -(
                self.assets["Cash"] - self.alpha * self.liabilities["Deposits"]
        )

        # Case disjunction: nothing to do is the repo_ask is negative
        if repo_ask <= 0.0:
            return

        # While loop over the list of banks that could accept to enter into a reverse repo with the instance self
        bank_list = list(self.trust.keys())
        while True:

            # algorithm to choose the bank to trade with using a multi-armed bandit protocol
            b = self.choose_bank(bank_list)
            bank_list.remove(b)
            rest = self.banks[b].enter_reverse_repo(self.id, repo_ask)
            self.update_learning(b, (repo_ask - rest) / repo_ask)

            # Test if the the bank agent owns enough collateral to enter into this repo
            assert (
                    self.assets["Securities Usable"] * self.collateral_value
                    + self.off_balance["Securities Collateral"] * self.collateral_value
                    - (repo_ask - rest)
                    > -float_limit
            ), self.__str__() + "\nNot Enough Collateral for bank {}".format(
                self.id
            )

            securities_usable_decrease = min(
                repo_ask - rest,
                self.assets["Securities Usable"] * self.collateral_value,
            )

            securities_collateral_decrease = max(
                repo_ask
                - rest
                - self.assets["Securities Usable"] * self.collateral_value,
                0.0,
            )

            # Fill-in the list of the repo amounts and steps history for the computation of the repo maturities
            if securities_usable_decrease > 0.0:
                self.repos_on_filo[b].append([securities_usable_decrease, self.steps])
            if securities_collateral_decrease > 0.0:
                self.repos_off_filo[b].append([securities_collateral_decrease, self.steps])

            # Update of the balance sheet items
            self.assets["Cash"] += repo_ask - rest
            self.on_balance_repos[b] += securities_usable_decrease
            self.assets["Securities Usable"] -= securities_usable_decrease / self.collateral_value
            self.assets["Securities Encumbered"] += (
                    securities_usable_decrease / self.collateral_value
            )
            self.off_balance_repos[b] += securities_collateral_decrease
            self.off_balance["Securities Collateral"] -= (
                    securities_collateral_decrease / self.collateral_value
            )
            self.off_balance["Securities Reused"] += (
                    securities_collateral_decrease / self.collateral_value
            )
            self.liabilities["Repos"] += repo_ask - rest
            repo_ask = rest
            if rest == 0.0 or len(bank_list) == 0:
                break

        # fix to handle the float error when accessing the repo market
        self.liabilities["MROs"] += repo_ask
        self.assets["Cash"] += repo_ask
        # for bank in bank_list:
        #     self.visits[bank] = self.visits[bank] - 0.1 * self.visits[bank]

    def choose_bank(self, bank_list):
        ucts = {}
        for b in self.trust.keys():
            if b in bank_list:
                ucts[b] = self.trust[
                              b
                          ] / self.steps + self.exploration_coeff * np.sqrt(
                    self.steps / self.visits[b]
                )
                # ucts[b] = self.trust[b] + self.exploration_coeff * np.sqrt(
                #     1.0 / self.visits[b]
                # )
            else:
                ucts[b] = 0.0
        return max(ucts, key=ucts.get)

    def update_learning(self, bank, value):
        self.visits[bank] += 1
        self.trust[bank] += value
        # self.visits[bank] = self.visits[bank] + 0.2 * (
        #     1.0 - self.visits[bank]
        # )
        # self.trust[bank] = self.trust[bank] + 0.2 * (value - self.trust[bank])

    def enter_reverse_repo(self, bank_id, amount):
        """
        Instance method allowing the instance of the ClassBank to enter into a reverse repo with bank bank_id,
        upon request of this latter. :param self: the bank bank_id requesting a repo.
        :param bank_id: the bank that requests the repo
        :param amount: the amount of cash requested.
        :return: the remaining amount of repo to the requested by the bank bank_id
        """

        # This is a fix on float error on excess of liquidity. A bank does not accept to enter into a reverse repo if
        # it has still repos, it means it is not in excess of cash.
        if (
                sum(self.on_balance_repos.values())
                + sum(self.off_balance_repos.values())
                > 0.0):
            return amount

        reverse_accept = max(
            self.assets["Cash"] - self.alpha * self.liabilities["Deposits"],
            0.0,
        )

        # Test if a bank is lending to itself due to a float error.
        assert self.id != bank_id, "Bank {} is lending to itself".format(
            self.id
        )

        # End the method if the amount that can be lent by the instance of ClassBank is negative.
        if reverse_accept <= 0.0:
            return amount

        # Update all balance sheet items related to the entering into a reverse repo
        reverse_repo = min(amount, reverse_accept)
        self.off_balance["Securities Collateral"] += reverse_repo / self.collateral_value
        self.assets["Cash"] -= reverse_repo
        self.assets["Reverse Repos"] += reverse_repo
        self.reverse_repos[bank_id] += reverse_repo

        # Return the remaining amount of repo to the requested by the bank bank_id
        return max(amount - reverse_accept, 0.0)

    # <editor-fold desc="Instance methods for the computation of regulatory ratios, and other balance sheet measures">
    def total_assets(self):
        """
        Instance method computing the total assets of an instance of ClassBank.
        :return:
        """
        return sum(self.assets.values())

    def total_liabilities(self):
        """
        Instance method computing the total liabilities of an instance of ClassBank.
        :return:
        """
        return sum(self.liabilities.values())

    def __str__(self):
        """
        Instance method allowing to print the full balance sheet of an instance of ClassBank.
        :return:
        """

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
            round(self.assets["Securities Usable"] * self.collateral_value, 2),
            round(
                self.assets["Securities Encumbered"] * self.collateral_value,
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
                self.off_balance["Securities Collateral"] * self.collateral_value,
                2,
            ),
            round(
                self.off_balance["Securities Reused"] * self.collateral_value,
                2,
            ),
            round(self.liquidity_coverage_ratio() * 100, 2),
            round(self.cash_to_deposits() * 100, 2),
            round(self.leverage_ratio() * 100, 2),
        )
        return p_str

    def liquidity_coverage_ratio(self):
        """
        Instance method computing the Liquidity Coverage Ratio (LCR) of an instance of ClassBank.
        :return:
        """
        lcr = (
                self.assets["Cash"]
                + self.assets["Securities Usable"] * self.collateral_value
        )
        lcr += self.off_balance["Securities Collateral"] * self.collateral_value
        lcr /= self.beta_reg * self.liabilities["Deposits"]
        return lcr

    def cash_to_deposits(self):
        """
        Instance method computing the cash-to-deposits ratio of an instance of ClassBank.
        :return:
        """
        return self.assets["Cash"] / (
                self.liabilities["Deposits"] + float_limit
        )

    def leverage_exposure(self):
        """
        Instance method computing the denominator of the leverage ratio of an instance of ClassBank. In case of
        absence of counterparty credit risk the leverage exposure is exactly the total assets. Here we do not model
        the bilateral agreements allowing to net short and long exposure between counterparties exchanging repos in
        both directions.
        :return:
        """
        ltsr = (
                self.assets["Cash"]
                + self.assets["Securities Usable"] * self.collateral_value
                + self.assets["Securities Encumbered"] * self.collateral_value
                + self.assets["Loans"]
                + self.assets["Reverse Repos"]

                # counterparty credit risk exposure, non zero only if the collateral value is dynamic
                + (
                        self.assets["Reverse Repos"]
                        - self.off_balance["Securities Collateral"] * self.collateral_value
                        - self.off_balance["Securities Reused"] * self.collateral_value
                )
        )
        return ltsr

    def leverage_ratio(self):
        """
        Instance method computing the leverage ratio of an instance of ClassBank.
        :return:
        """
        return self.liabilities["Own Funds"] / self.leverage_exposure()

    def assert_lcr(self):
        """
        Instance method evaluating if the LCR constraint is meet for an instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and time step concerned.
        """
        assert (
                self.assets["Cash"]
                + self.off_balance["Securities Collateral"] * self.collateral_value
                + self.assets["Securities Usable"] * self.collateral_value
                - self.liabilities["Deposits"] * self.beta_reg
                >= -float_limit
        ), self.__str__() + "\nLCR not at its target value for bank {} at step {}".format(
            self.id, self.steps
        )

    def assert_minimum_reserves(self):
        """
        Instance method evaluating if the minimum reserves constraint is meet for an instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and time step concerned.
        """
        assert (
                self.assets["Cash"] - self.alpha * self.liabilities["Deposits"]
                >= -float_limit
        ), self.__str__() + "\nMinimum reserves not respected for bank {} at step {}".format(
            self.id, self.steps
        )

    def assert_leverage(self):
        """
        Instance method evaluating if the leverage ratio constraint is meet for an instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and time step concerned.
        """
        assert (
                self.liabilities["Own Funds"]
                - self.gamma * self.leverage_exposure()
                > -float_limit
        ), self.__str__() + "\nLeverage ratio below its regulatory requirement for bank {} at step {}" "".format(
            self.id, self.steps
        )

    def assert_alm(self):
        """
        Instance method evaluating if accounting constraints are meet for an instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and time step concerned.
        """
        assert (
                np.abs(self.total_assets() - self.total_liabilities())
                < float_limit
        ), self.__str__() + "\nAssets don't match Liabilities for bank {} at step {}".format(
            self.id, self.steps
        )
    # </editor-fold>
