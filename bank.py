import os

# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import parameters as par
import pickle


class ClassBank:
    # instance of Class Banks">
    def __init__(
        self,
        Network,  # pointeur vers network (pour éviter les boucle for dans network)
        id,
        initial_deposits,
        alpha_init,
        alpha,
        beta_init,
        beta_reg,
        beta_star,
        gamma,
        nb_banks,
        collateral_value,
        conservative_shock,
        LCR_mgt_opt,
    ):
        # Initialisation of the class parameters.
        self.Network = Network
        self.id = id
        self.alpha = alpha
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.beta_reg = beta_reg
        self.beta_star = beta_star
        self.gamma = gamma
        self.collateral_value = collateral_value
        self.conservative_shock = conservative_shock
        self.LCR_mgt_opt = LCR_mgt_opt
        self.nb_banks = nb_banks
        self.initial_deposits = initial_deposits

        # definition of a dictionary (faster) with the accounting data
        self.dic_balance_sheet = dict.fromkeys(par.accounting_items, 0)

        # definition of the df for storing all reverse repo transactions
        self.df_rev_repo_trans = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [],
                names=["bank_id", "trans_id"],
            ),
            columns=par.transaction_cols,
        )

        # Other parameters initialization
        self.excess_liquidity = (
            0.0  # Excess liquidity a bank can provide in a reverse repo.
        )
        self.shock = (
            0.0  # Value of the shock received by a bank at a given time_step.
        )

        # Parameters related to the learning algorithm for the selection of counterparties on the repo market
        self.trust = (
            {}
        )  # Dictionary of the trust coefficients with the other banks.

        # Dictionaries & lists related to the other bank agents
        self.on_repo_exp = (
            {}
        )  # Dictionary of the repo exposures  received from each of the other banks using securities.
        self.off_repo_exp = (
            {}
        )  # Dictionary of the repo exposures received from each of the other banks, re-using collateral.
        self.rev_repo_exp = (
            {}
        )  # Dictionary of the reverse repo exposures provided to each of the other banks.
        self.banks = (
            {}
        )  # Dictionary of the instances of the ClassBank class existing in
        # the ClassNetwork class.

        # Run each of the
        self.initialize_balance_sheet()

    def initialize_balance_sheet(self):

        # set the initial deposits
        self.dic_balance_sheet["deposits"] = self.initial_deposits

        # The cash is set to its minimum reserve amount.
        self.dic_balance_sheet["cash"] = (
            self.dic_balance_sheet["deposits"] * self.alpha_init
        )

        # The collateral is set to the amount allowing to match the beta_init.
        self.dic_balance_sheet["securities usable"] = (
            (self.beta_init - self.alpha)
            * self.dic_balance_sheet["deposits"]
            / self.collateral_value
        )

        # The Own-funds are set to match the leverage ratio.
        self.dic_balance_sheet["own funds"] = (
            self.gamma / (1.0 - self.gamma)
        ) * self.dic_balance_sheet["deposits"]

        # The loans are set to match the assets and liabilities.
        self.dic_balance_sheet["loans"] = (
            self.dic_balance_sheet["own funds"]
            + self.dic_balance_sheet["deposits"]
            - self.dic_balance_sheet["cash"]
            - self.dic_balance_sheet["securities usable"]
            * self.collateral_value
        )

    def initialize_banks(self, banks):
        """
        Initialize the list of dictionaries of the bank agent instance self.
        with the information about each of the
        other banks in the network.
        :param banks : a list of ClassBank class instance, it is a parameter
        of the ClassNetwork class.
        :return:
        """
        for bank in banks:
            self.rev_repo_exp[bank.id] = 0.0
            if bank.id != self.id:
                self.trust[bank.id] = 0.0
                self.on_repo_exp[bank.id] = 0.0
                self.off_repo_exp[bank.id] = 0.0
                self.banks[bank.id] = bank

    def step_set_shock(self, shock):
        self.dic_balance_sheet["deposits"] += shock
        self.dic_balance_sheet["cash"] += shock

    def set_collateral(self, collateral_value):
        """
        Instance method updating the value of the collateral of an instance
        of ClassBank.
        :param collateral_value:
        :param shock: shock applied
        :return:
        """
        self.collateral_value = collateral_value

    def step_lcr_mgt(self):
        """
        Instance method updating the cash, loans, and MROs of an instance of
        ClassBank.
        """
        # Definition of the variation of cash required to match the targeted
        # LCR, this value can be positive or
        # negative.
        delta_cash = (
            self.beta_star * self.dic_balance_sheet["deposits"]
            - self.dic_balance_sheet["cash"]
            - self.dic_balance_sheet["securities usable"]
            * self.collateral_value
            - self.dic_balance_sheet["securities collateral"]
            * self.collateral_value
        )

        # Fill-in the Cash, Loans, and Main Refinancing Operations (MROs).
        # In case of a negative delta cash,
        # the bank first reimburses its existing central bank funding (MRO)
        # before granting new loans.
        self.dic_balance_sheet["cash"] += delta_cash
        self.dic_balance_sheet["loans"] += -min(
            self.dic_balance_sheet["central bank funding"] + delta_cash, 0.0
        )
        self.dic_balance_sheet["central bank funding"] = max(
            self.dic_balance_sheet["central bank funding"] + delta_cash, 0.0
        )

    def step_end_repos(self):
        """
        Instance method to perform the step of ending repos. It actually
        calls the end_repos method which can be used
        either to end repo due to excess liquidity or due to collateral needs.
        :return:
        """
        excess_liquidity = (
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
        )
        excess_liquidity = max(excess_liquidity, 0.0)
        self.end_repos(excess_liquidity)

    def end_repos(self, target_repo_amount_to_close):
        """
        Instance method allows to close all the repos from an instance of
        ClassBank, as well as all the necessary
        repos of the lenders of cash to this bank agent, in a recursive manner.
        :param: repo_amount: amount of excess_liquidity or collateral to
        call back in case of chain of repos ending.
        :return:
        """

        # stop the method if the mount of repo to close is negative
        if target_repo_amount_to_close <= 0.0:
            return

        # 1) -------------------------------------------------------------
        # First step, end all off-balance repos, starting by the ones with
        # banks having the lowest trust factor.
        trust = self.trust.copy()

        # For loop over the sorted list of the other banks, starting by the
        # lowest trust factor.
        for b, t in sorted(
            trust.items(), key=lambda item: item[1], reverse=False
        ):
            # Definition of the target amount of off-balance repos to close
            # with a given bank b
            end = min(self.off_repo_exp[b], target_repo_amount_to_close)

            if self.off_repo_exp[b] - end < -par.float_limit:
                print(self.off_repo_exp[b])

            if (
                self.off_repo_exp[b]
                - self.dic_balance_sheet["securities reused"]
                > par.float_limit
            ):
                print(
                    f"The end repo amount of bank {self.id} with bank {b} is {end} while the off_balance_repo is {self.off_repo_exp[b]} and the securities reused is {self.dic_balance_sheet['securities reused']}"
                )

            # test the accuracy of self.on_balance repos:
            assert (
                abs(
                    sum(self.off_repo_exp.values())
                    - self.dic_balance_sheet["securities reused"]
                )
                < par.float_limit
            ), (
                "the sum of the "
                "off-balance repos {}, "
                "is not equal to the "
                "Securities Reused {}".format(
                    sum(self.off_repo_exp.values()),
                    self.dic_balance_sheet["securities reused"],
                )
            )

            # Case disjunction
            if end > 0.0:

                # The bank id b must close its reverse repo with the bank agent self, the instance method end_reverse_repo calls recursively the end_repos method for the bank b in case this latter would not own sufficient colletral to end its reverse repo.

                self.banks[b].end_reverse_repo(self.id, end)

                # Update all the balance sheet items related to the closure of the repo
                self.dic_balance_sheet["cash"] -= end
                self.dic_balance_sheet["repo balance"] -= end
                self.dic_balance_sheet["securities collateral"] += (
                    end / self.collateral_value
                )
                self.dic_balance_sheet["securities reused"] -= (
                    end / self.collateral_value
                )

                self.off_repo_exp[b] -= end

                # # when the off balance repo is below some limit, we set the value of the exposure to 0
                # if self.off_repo_exp[b] < par.float_limit:
                #     self.off_repo_exp[b] = 0

                target_repo_amount_to_close -= end

                assert (
                    self.dic_balance_sheet["securities reused"]
                    >= -par.float_limit
                ), (
                    "securities reused negative {} at step {}, due to "
                    "retrieving of end {}".format(
                        self.dic_balance_sheet["securities reused"],
                        self.Network.step,
                        end,
                    )
                )

            # Break the for loop if off balance repos are sufficient to
            # empty the excess liquidity
            if target_repo_amount_to_close == 0.0:
                break

        # End the instance method if off balance repos are sufficient to
        # empty the excess liquidity
        if target_repo_amount_to_close == 0.0:
            return

        # 2) -------------------------------------------------------------
        # Second step, end all on-balance repos, starting by the ones with
        # banks having the lowest trust factor.
        for b, t in sorted(
            trust.items(), key=lambda item: item[1], reverse=True
        ):
            end = min(self.on_repo_exp[b], target_repo_amount_to_close)

            # test the accuracy of self.on_balance repos:
            assert (
                abs(
                    sum(self.on_repo_exp.values())
                    - self.dic_balance_sheet["securities encumbered"]
                )
                < par.float_limit
            ), f"the sum of the on-balance repos {sum(self.on_repo_exp.values())} are not equal to the Securities Encumbered {self.dic_balance_sheet['securities encumbered']}"

            if end > 0.0:

                # The bank id b must close its reverse repo with the bank
                # agent self, the instance method
                # end_reverse_repo calls recursively the end_repos method
                # for the bank b in case this latter would
                # not own sufficient colletral to end its reverse repo.
                self.banks[b].end_reverse_repo(self.id, end)

                self.dic_balance_sheet["cash"] -= end
                self.dic_balance_sheet["repo balance"] -= end
                self.dic_balance_sheet["securities usable"] += (
                    end / self.collateral_value
                )
                self.dic_balance_sheet["securities encumbered"] -= (
                    end / self.collateral_value
                )
                self.on_repo_exp[b] -= end

                # # when the on balance repo is below some limit, we set the value of the exposure to 0
                # if self.on_repo_exp[b] < par.float_limit:
                #     self.on_repo_exp[b] = 0

                # fix, it seems the previous version was an error
                target_repo_amount_to_close -= end

            # Break the for loop if on balance repos are sufficient to empty
            # the excess liquidity
            if target_repo_amount_to_close == 0.0:
                break

    def end_reverse_repo(self, bank_id, amount):
        """
        Instance method allowing a instance of the ClassBank self, to close
        its reverse repo with the bank bank_id,
        upon request of of this latter. This instance method calls
        recursively the end_repos method with the missing
        amount of collateral to be provided back.
        :param bank_id: identifier of the bank requesting to close its
        repo for a given amount.
        :param amount: amount requested to be closed.
        :return:
        """

        if amount <= 0.0:
            return

        # Definition of the amount of collateral missing to allow the
        # closing of the reverse repo. We do not allow the bank
        # self to use its securities usable as the substitute for the
        # securities he received in the first place.
        missing_collateral = max(
            amount
            - self.dic_balance_sheet["securities collateral"]
            * self.collateral_value,
            0.0,
        )

        # Recursive calling of the missing amount of collateral through the
        # ending of the own repos of the bank agent.
        self.end_repos(missing_collateral)

        # Recompute the missing amount of collateral after the end of the
        # recursive algorithm.
        missing_collateral = max(
            amount
            - self.dic_balance_sheet["securities collateral"]
            * self.collateral_value,
            0.0,
        )

        # Assert if the recursive algorithm worked adequately, otherwise
        # print an error
        assert missing_collateral <= par.float_limit, (
            self.__str__() + "\nBank {} has not enough collateral to end "
            "its reverse repo with bank {}, missing "
            "amount is {}".format(
                self.id,
                bank_id,
                missing_collateral,
            )
        )

        # Update all the required balance sheet items by the closing of the
        # reverse repo. The bank self use only its securities collateral to
        # end its reverse repo.

        assert not (
            self.dic_balance_sheet["securities reused"] > par.float_limit
            and self.dic_balance_sheet["securities usable"] > par.float_limit
        ), (
            "both reused {} and "
            "usable {} "
            "are positive, "
            "while normally supposed to use all usable before using "
            "collateral".format(
                self.dic_balance_sheet["securities reused"],
                self.dic_balance_sheet["securities usable"],
            )
        )

        assert (
            abs(
                self.dic_balance_sheet["securities collateral"]
                + self.dic_balance_sheet["securities reused"]
                - self.dic_balance_sheet["reverse repo balance"]
            )
            < par.float_limit
        ), (
            "incorrect balance sheet \n securities collateral {},"
            "\n "
            "securities reused {}"
            "\n "
            "reverse "
            "repos {} "
            ""
            "\n "
            "difference {}"
            "".format(
                self.dic_balance_sheet["securities collateral"],
                self.dic_balance_sheet["securities reused"],
                self.dic_balance_sheet["reverse repo balance"],
                self.dic_balance_sheet["reverse repo balance"]
                - self.dic_balance_sheet["securities collateral"]
                - self.dic_balance_sheet["securities reused"],
            )
        )

        # Update of the balance sheet items
        self.dic_balance_sheet["cash"] += amount
        self.dic_balance_sheet["reverse repo balance"] -= amount
        self.rev_repo_exp[bank_id] -= amount
        self.dic_balance_sheet["securities collateral"] -= (
            amount / self.collateral_value
        )

        # update df_reverse_repos
        # initialize the list of trans_ids with status == True
        trans_ids = self.df_rev_repo_trans.loc[bank_id][
            self.df_rev_repo_trans.loc[bank_id]["status"]
        ].index

        # case with an issue (no solution yet, just print the issue and a warning)
        if len(trans_ids) == 0:
            self.df_rev_repo_trans.to_csv(
                f"./support/errors/{self.id}_df_reverse_repo_err.csv"
            )
            pickle.dump(
                self.banks[bank_id].on_repo_exp,
                open(
                    f"./support/errors/{bank_id}_on_balance_repos.pickle", "wb"
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            pickle.dump(
                self.banks[bank_id].off_repo_exp,
                open(
                    f"./support/errors/{bank_id}_off_balance_repos.pickle",
                    "wb",
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            print(
                "WARNING: df_reverse_repo and on_balance_repo/off_balance_repos do not match !"
            )

        if len(trans_ids) > 0:
            trans_id = trans_ids[0]
            last_trans_id = trans_ids[-1]
            remaining_amount = amount
            i = 0
            while (
                remaining_amount
                >= self.df_rev_repo_trans.loc[(bank_id, trans_id), "amount"]
            ):
                # close the transaction
                self.df_rev_repo_trans.loc[
                    (bank_id, trans_id), "status"
                ] = False

                # record the step of the closing of the transaction
                self.df_rev_repo_trans.loc[(bank_id, trans_id), "end_step"] = self.Network.step
                    
                # decrease the remaining amount
                remaining_amount -= self.df_rev_repo_trans.loc[
                    (bank_id, trans_id), "amount"
                ]

                # next transaction
                if trans_id < last_trans_id:
                    i += 1
                    trans_id = trans_ids[i]

            # specific case if the remaining amount is smaller than the transaction size: need to create and close a special transaction
            if (
                remaining_amount
                < self.df_rev_repo_trans.loc[(bank_id, trans_id), "amount"]
            ) and (remaining_amount>self.Network.min_repo_trans_size):

                # create a new transaction with the status closed
                start_step = self.df_rev_repo_trans.loc[
                    (bank_id, trans_id), "start_step"
                ]
                self.df_rev_repo_trans.loc[(bank_id, last_trans_id + 1), :] = [
                    remaining_amount,
                    start_step,
                    self.Network.step,
                    False,
                ]

                # update the amount of the transaction
                self.df_rev_repo_trans.loc[
                    (bank_id, trans_id), "amount"
                ] -= remaining_amount

        # Once a reverse repo is ended, there is an excess liquidity which
        # requires closing the own repos of the bank self
        self.step_end_repos()

    def step_enter_repos(self):
        """
        Instance method allowing an instance of ClassBank to enter into a
        series of repo to meet its liquidity needs.
        """

        # Define the amount of repo to be requested (it is a positive amount
        # if there is a liquidity need)
        repo_ask = -(
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
        )

        # if there is no LCR mgt (no ECB funding to mgt LCR), we might have a bigger shock to absorb than the available collateral, so only a part of the shock is absorded on the repo market
        if not (self.LCR_mgt_opt):
            repo_ask = min(
                repo_ask,
                self.dic_balance_sheet["securities usable"]
                * self.collateral_value
                + self.dic_balance_sheet["securities collateral"]
                * self.collateral_value,
            )

        # Case disjunction: nothing to do if the repo_ask is negative
        if repo_ask <= 0.0:
            return

        # While loop over the list of banks that could accept to enter into
        # a reverse repo with the instance self
        bank_list = list(self.trust.keys())
        while True:

            # algorithm to choose the bank to trade with
            b = self.choose_bank(bank_list)
            bank_list.remove(b)
            rest = self.banks[b].enter_reverse_repo(self.id, repo_ask)
            self.update_learning(b, (repo_ask - rest) / repo_ask)

            # Test if the bank agent owns enough collateral to enter
            # into this repo
            assert (
                self.dic_balance_sheet["securities usable"]
                * self.collateral_value
                + self.dic_balance_sheet["securities collateral"]
                * self.collateral_value
                - (repo_ask - rest)
                > -par.float_limit
            ), f"{self.__str__()}\nNot Enough Collateral for bank {self.id}"

            securities_usable_decrease = min(
                repo_ask - rest,
                self.dic_balance_sheet["securities usable"]
                * self.collateral_value,
            )

            securities_collateral_decrease = max(
                repo_ask
                - rest
                - self.dic_balance_sheet["securities usable"]
                * self.collateral_value,
                0.0,
            )

            # Update of the balance sheet items
            self.dic_balance_sheet["cash"] += repo_ask - rest
            self.on_repo_exp[b] += securities_usable_decrease
            self.dic_balance_sheet["securities usable"] -= (
                securities_usable_decrease / self.collateral_value
            )
            self.dic_balance_sheet["securities encumbered"] += (
                securities_usable_decrease / self.collateral_value
            )
            self.off_repo_exp[b] += securities_collateral_decrease
            self.dic_balance_sheet["securities collateral"] -= (
                securities_collateral_decrease / self.collateral_value
            )
            self.dic_balance_sheet["securities reused"] += (
                securities_collateral_decrease / self.collateral_value
            )
            self.dic_balance_sheet["repo balance"] += repo_ask - rest

            assert not (
                self.dic_balance_sheet["securities reused"] > par.float_limit
                and self.dic_balance_sheet["securities usable"]
                > par.float_limit
            ), (
                "both reused {} and "
                "usable {} "
                "are positive, "
                "while normally "
                "supposed to use all "
                "usable before using "
                "collat".format(
                    self.dic_balance_sheet["securities reused"],
                    self.dic_balance_sheet["securities usable"],
                )
            )

            repo_ask = rest
            if rest <= self.Network.min_repo_trans_size or len(bank_list) == 0:
                break

        # check for errors, in case of conservative shocks and LCR mgt, all repo request should be satisfied
        if self.LCR_mgt_opt and self.conservative_shock:
            if repo_ask > par.float_limit:
                for b in self.banks.keys():
                    print(self.banks[str(b)])
            assert repo_ask <= par.float_limit, (
                "repo request unsatified for bank {},"
                " for the amount {}".format(self.id, repo_ask)
            )

    def step_central_bank_funding(self):
        """
        In case shocks are non conversative banks may request cash to the central bank as a last resort when there is not enough cash available on the repo market.
        In case of absence of CB funding for LCR management, banks may request cash to the central bank as a last resort when there is not enough collateral available for performing repos.
        """

        # Define the amount of repo to be requested (it is a positive amount
        # if there is a liquidity need)
        MRO_ask = -(
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
        )

        # Case disjunction: nothing to do if the repo_ask is negative
        if MRO_ask <= 0.0:
            return

        else:
            # perform a central bank funding
            self.dic_balance_sheet["central bank funding"] += MRO_ask
            self.dic_balance_sheet["cash"] += MRO_ask

    def choose_bank(self, bank_list):
        trusts = {}
        for b in bank_list:
            trusts[b] = self.trust[b]
        return max(trusts, key=trusts.get)

    def update_learning(self, bank, value):
        # Lux's approach: creates a trust between 0 and 1, when value is between 0 and 1, deacing in power 2 toward 0 in case value is 0, converging in power 2 toward 1 in case value is 1
        self.trust[bank] = self.trust[bank] + 0.5 * (value - self.trust[bank])

    def enter_reverse_repo(self, bank_id, amount):
        """
        Instance method allowing the instance of the ClassBank to enter into
        a reverse repo with bank bank_id,
        upon request of this latter.
        :param self: the bank bank_id requesting a repo.
        :param bank_id: the bank that requests the repo
        :param amount: the amount of cash requested.
        :return: the remaining amount of repo to the requested by the bank
        bank_id
        """

        #  A bank does not accept to enter into a reverse repo if
        # it has still repos, it means it is not in excess of cash.
        if (
            sum(self.on_repo_exp.values()) + sum(self.off_repo_exp.values())
            > 0.0
        ):
            return amount

        reverse_accept = max(
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"],
            0.0,
        )

        # Test if a bank is lending to itself due to a float error.
        assert self.id != bank_id, "Bank {} is lending to itself".format(
            self.id
        )

        # End the method if the amount that can be lent by the instance of ClassBank is lower than the min_rev_repo_size.
        if reverse_accept <= self.Network.min_repo_trans_size:
            return amount

        # Update all balance sheet items related to the entering into a
        # reverse repo
        reverse_repo = min(amount, reverse_accept)
        self.dic_balance_sheet["securities collateral"] += (
            reverse_repo / self.collateral_value
        )
        self.dic_balance_sheet["cash"] -= reverse_repo
        self.dic_balance_sheet["reverse repo balance"] += reverse_repo
        self.rev_repo_exp[bank_id] += reverse_repo

        # fill df_reverse_repos
        if bank_id in self.df_rev_repo_trans.index.get_level_values("bank_id"):
            trans_id = self.df_rev_repo_trans.loc[bank_id].index[-1] + 1
        else:
            trans_id = 0
        self.df_rev_repo_trans.loc[(bank_id, trans_id), :] = [
            reverse_repo,
            self.Network.step,
            np.NaN,
            True,
        ]

        # Return the remaining amount of repo to the requested by the bank
        # bank_id
        return max(amount - reverse_accept, 0.0)

    def total_assets(self):
        """
        Instance method computing the total assets of an instance of ClassBank.
        :return:
        """
        return sum(self.dic_balance_sheet.values())

    def total_liabilities(self):
        """
        Instance method computing the total liabilities of an instance of
        ClassBank.
        :return:
        """
        return sum(self.dic_balance_sheet.values())

    def __str__(self):
        """
        Instance method allowing to print the full balance sheet of an
        instance of ClassBank.
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
            round(self.dic_balance_sheet["cash"], 2),
            round(
                self.dic_balance_sheet["securities usable"]
                * self.collateral_value,
                2,
            ),
            round(
                self.dic_balance_sheet["securities encumbered"]
                * self.collateral_value,
                2,
            ),
            round(self.dic_balance_sheet["reverse repo balance"], 2),
            round(self.dic_balance_sheet["loans"], 2),
            round(self.total_liabilities(), 2),
            round(self.dic_balance_sheet["own funds"], 2),
            round(self.dic_balance_sheet["deposits"], 2),
            round(self.dic_balance_sheet["repo balance"], 2),
            round(self.dic_balance_sheet["central bank funding"], 2),
            round(
                self.dic_balance_sheet["securities collateral"]
                * self.collateral_value,
                2,
            ),
            round(
                self.dic_balance_sheet["securities reused"]
                * self.collateral_value,
                2,
            ),
            round(self.liquidity_coverage_ratio() * 100, 2),
            round(self.cash_to_deposits() * 100, 2),
            round(self.leverage_ratio() * 100, 2),
        )
        return p_str

    def liquidity_coverage_ratio(self):
        """
        Instance method computing the Liquidity Coverage Ratio (LCR) of an
        instance of ClassBank.
        :return:
        """
        lcr = (
            self.dic_balance_sheet["cash"]
            + self.dic_balance_sheet["securities usable"]
            * self.collateral_value
        )
        lcr += (
            self.dic_balance_sheet["securities collateral"]
            * self.collateral_value
        )
        lcr /= self.beta_reg * self.dic_balance_sheet["deposits"]
        return lcr

    def cash_to_deposits(self):
        """
        Instance method computing the cash-to-deposits ratio of an instance
        of ClassBank.
        :return:
        """
        return self.dic_balance_sheet["cash"] / (
            self.dic_balance_sheet["deposits"]
        )

    def leverage_exposure(self):
        """
        Instance method computing the denominator of the leverage ratio of
        an instance of ClassBank. In case of
        absence of counterparty credit risk the leverage exposure is exactly
        the total assets. Here we do not model
        the bilateral agreements allowing to net short and long exposure
        between counterparties exchanging repos in
        both directions.
        :return:
        """
        ltsr = (
            self.dic_balance_sheet["cash"]
            + self.dic_balance_sheet["securities usable"]
            * self.collateral_value
            + self.dic_balance_sheet["securities encumbered"]
            * self.collateral_value
            + self.dic_balance_sheet["loans"]
            + self.dic_balance_sheet["reverse repo balance"]
            # counterparty credit risk exposure, non zero only if the
            # collateral value is dynamic
            + (
                self.dic_balance_sheet["reverse repo balance"]
                - self.dic_balance_sheet["securities collateral"]
                * self.collateral_value
                - self.dic_balance_sheet["securities reused"]
                * self.collateral_value
            )
        )
        return ltsr

    def leverage_ratio(self):
        """
        Instance method computing the leverage ratio of an instance of
        ClassBank.
        :return:
        """
        return self.dic_balance_sheet["own funds"] / self.leverage_exposure()

    def check_balance_sheet(self):
        """
        Instance method evaluating if accounting constraints are meet for an
        instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and
        time step concerned.
        """
        assert (
            np.abs(self.total_assets() - self.total_liabilities())
            < par.float_limit
        ), self.__str__() + "\nAssets don't match Liabilities for bank {} at " "step {}, for the amount {}".format(
            self.id,
            self.Network.step,
            (self.total_assets() - self.total_liabilities()),
        )

    def check_min_reserves(self):
        """
        Instance method evaluating if the minimum reserves constraint is
        meet for an instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and
        time step concerned.
        """

        # in case the minimum reserve is not respected, print all the positions of the banks in the network
        if (
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
            < -par.float_limit
        ):
            for bank in self.banks.values():
                print(bank)
                print(bank.reverse_repo_exposures)

        assert (
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
            >= -par.float_limit
        ), (
            self.__str__() + "\nMinimum reserves not respected for bank {} at"
            " "
            "step {} \n The reverse repos provided to the rest of the network are {}".format(
                self.id, self.Network.step, self.rev_repo_exp
            )
        )

    def check_lcr(self):
        """
        Instance method evaluating if the LCR constraint is meet for an
        instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and
        time step concerned.
        """
        assert (
            self.dic_balance_sheet["cash"]
            + self.dic_balance_sheet["securities collateral"]
            * self.collateral_value
            + self.dic_balance_sheet["securities usable"]
            * self.collateral_value
            - self.dic_balance_sheet["deposits"] * self.beta_reg
            >= -par.float_limit
        ), self.__str__() + "\nLCR not at its target value for bank {} at " "step {}".format(
            self.id, self.Network.step
        )

    def assert_leverage(self):
        """
        Instance method evaluating if the leverage ratio constraint is meet
        for an instance of ClassBank.
        :return: Breaks the code and returns a description of the bank and
        time step concerned.
        """
        assert (
            self.dic_balance_sheet["own funds"]
            - self.gamma * self.leverage_exposure()
            > -par.float_limit
        ), (
            self.__str__() + "\nLeverage ratio below its regulatory "
            "requirement for bank {} at step {}"
            "".format(self.id, self.Network.step)
        )

    def store_df_reverse_repos(self, path):
        self.df_rev_repo_trans.to_csv(f"{path}df_reverse_repos.csv")
