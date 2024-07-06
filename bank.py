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
        gamma_init,
        gamma,
        collateral_value,
    ):
        # Initialisation of the class parameters.
        self.Network = Network
        self.id = id
        self.alpha = alpha
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.beta_reg = beta_reg
        self.beta_star = beta_star
        self.gamma_init = gamma_init
        self.gamma = gamma
        self.collateral_value = collateral_value
        self.initial_deposits = initial_deposits
        

        # definition of a dictionary (faster) with the accounting data
        self.dic_balance_sheet = dict.fromkeys(par.accounting_items, 0)
        self.ar_cbfund_trans = np.zeros(1000) #temp

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

        self.dic_loans_steps_closing = {} # dictionary associating the the step at which loans will be closed to the closing amount

        # Run each of the
        self.initialize_balance_sheet()

    def initialize_balance_sheet(self):

        # set the initial deposits
        self.dic_balance_sheet["deposits"] = self.initial_deposits

        # set the initial cash
        if self.alpha_init:
            # The cash is set to its minimum reserve amount.
            self.dic_balance_sheet["cash"] = (
                self.dic_balance_sheet["deposits"] * self.alpha_init
            )
        else:
            #  The cash is set to 0 (because it cannot be created by banks !)
            self.dic_balance_sheet["cash"] = 0


        # The collateral is set to the amount allowing to match the beta_init.
        if self.alpha_init:
            self.dic_balance_sheet["securities usable"] = (
            (self.beta_init - self.alpha)
            * self.dic_balance_sheet["deposits"]
            / self.collateral_value
            )
        else:
            self.dic_balance_sheet["securities usable"] = self.beta_init * self.dic_balance_sheet["deposits"] / self.collateral_value

        # The Own-funds are set to match the leverage ratio. gamma_init
        self.dic_balance_sheet["own funds"] = (
            self.gamma_init / (1.0 - self.gamma_init)
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
        
        ar_init_trust = np.random.uniform(0,1,self.Network.nb_banks)
        
        for bank in banks:
            self.rev_repo_exp[bank.id] = 0.0
            if bank.id != self.id:
                
                if not(self.Network.initial_deposits_size):
                    self.trust[bank.id] = ar_init_trust[bank.id]
                    
                else:
                    self.trust[bank.id] = self.Network.df_banks.loc[bank.id,"total assets"] / self.Network.df_banks["total assets"].sum()
                 
                self.on_repo_exp[bank.id] = 0.0
                self.off_repo_exp[bank.id] = 0.0
                self.banks[bank.id] = bank

    def set_shock(self, shock):
        self.dic_balance_sheet["deposits"] += shock
        self.dic_balance_sheet["cash"] += shock

    def set_money_creation(self,new_money):

        if self.Network.gamma_new:

            # we assume an economic agent recieves an infinit lenght loan and invest it in the capital of the bank 
            new_own_funds = self.Network.gamma_new*new_money
            self.dic_balance_sheet["loans"] += new_own_funds
            self.dic_balance_sheet["own funds"] += new_own_funds
            new_money -= new_own_funds

        if self.Network.beta_new:
            new_securities = new_money*self.Network.beta_new

            if self.Network.step >= self.Network.QE_start and self.Network.step < self.Network.QE_stop:
                # the central bank buys the bonds issued by the government and the governement deposits the cash in the banking system
                self.dic_balance_sheet["cash"] += new_securities
                self.dic_balance_sheet["deposits"] += new_securities
                
            else:
                # we assume the bank buys a bond issued by the governement 
                self.dic_balance_sheet["securities usable"] += new_securities
                self.dic_balance_sheet["deposits"] += new_securities
                new_money -= new_securities
            
        # The remaining amount is a loan with a fixed maturity
        self.dic_balance_sheet["loans"] += new_money
        self.dic_balance_sheet["deposits"] += new_money
        self.dic_loans_steps_closing.update({self.Network.step+self.Network.loan_tenor:new_money})

    def lcr_mgt(self):
        """
        Instance method updating the cash and cb funding of an instance of
        ClassBank.
        """
        # Definition of the variation of cash required to minimize the LCR, this value can be positive or negative.
        delta_cash = (
            self.beta_star * self.dic_balance_sheet["deposits"]
            - self.dic_balance_sheet["cash"]
            - self.dic_balance_sheet["securities usable"]
            * self.collateral_value
            - self.dic_balance_sheet["securities collateral"]
            * self.collateral_value
        )

        # Fill-in the Cash and central bank funding
        delta_amount  = max(
            self.dic_balance_sheet["central bank funding"] + delta_cash, 0.0
        ) - self.dic_balance_sheet["central bank funding"]
        self.dic_balance_sheet["cash"] += delta_amount
        self.dic_balance_sheet["central bank funding"] += delta_amount

    def leverage_mgt(self):
        
        # we can deleverage either by end repos or central bank funding but the later would kill our LCR
        if self.dic_balance_sheet["own funds"] / self.leverage_exposure() < self.Network.gamma_star:
            self.step_end_repos()

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
            # Definition of the target amount of off-balance repos to close with a given bank b
            end = min(self.off_repo_exp[b], target_repo_amount_to_close)
            
            if (
                self.off_repo_exp[b]
                - self.dic_balance_sheet["securities reused"]
                > par.float_limit
            ):
                print(f"The end repo amount of bank {self.id} with bank {b} is {end} while the off_balance_repo is {self.off_repo_exp[b]} and the securities reused is {self.dic_balance_sheet['securities reused']}")

            # test the accuracy of self.on_balance repos:
            assessment  = (abs(
                    sum(self.off_repo_exp.values())
                    - self.dic_balance_sheet["securities reused"]) < par.float_limit)
            if not(assessment):
                self.Network.str_output_error = f"***ERROR***: the sum of the off-balance repos {sum(self.off_repo_exp.values())} is not equal to the Securities Reused {self.dic_balance_sheet['securities reused']}. Plot and stop simulation."
                print(self.Network.str_output_error)

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

                # # update the chains if the remaining amount is negligeable (only for repo with reuse)
                # if self.off_repo_exp[b] < par.float_limit:
                #     self.Network.remove_rev_repo_from_chains(b,self.id)

                # # when the off balance repo is below some limit, we set the value of the exposure to 0
                # if self.off_repo_exp[b] < par.float_limit:
                #     self.off_repo_exp[b] = 0

                target_repo_amount_to_close -= end

                assessment =  (
                    self.dic_balance_sheet["securities reused"]
                    >= -par.float_limit
                )
                if not(assessment):
                    self.Network.str_output_error = f"***ERROR***: securities reused negative {self.dic_balance_sheet['securities reused']} at step {self.Network.step}, due to retrieving of end {end}. Plot and stop simulation."
                    print(self.Network.str_output_error)

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
            assessment=  (
                abs(
                    sum(self.on_repo_exp.values())
                    - self.dic_balance_sheet["securities encumbered"]
                )
                < par.float_limit
            )
            
            if not(assessment):
                    self.Network.str_output_error = f"***ERROR***: the sum of the on-balance repos {sum(self.on_repo_exp.values())} are not equal to the Securities Encumbered {self.dic_balance_sheet['securities encumbered']}. Plot and stop simulation."
                    print(self.Network.str_output_error)

            if end > 0.0:

                # The bank id b must close its reverse repo with the bank agent self, the instance method end_reverse_repo calls recursively the end_repos method for the bank b in case this latter would not own sufficient colletral to end its reverse repo.
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
        its reverse repo with the bank bank_id upon request of of this latter. This instance method calls recursively the end_repos method with the missing amount of collateral to be provided back.
        :param bank_id: identifier of the bank requesting to close its
        repo for a given amount.
        :param amount: amount requested to be closed.
        """

        if amount <= 0.0:
            return

        # Definition of the amount of collateral missing to allow the closing of the reverse repo.
        if self.Network.substitution:
            #  We allow the bank self to use its securities usable as substitute for the securities she received in the first place
            missing_collateral = max(
                amount
                - self.dic_balance_sheet["securities collateral"] - self.dic_balance_sheet["securities usable"],
                0.0,
            )
        
        else:
            # otherwise only the securities collateral can be used
            missing_collateral = max(
                amount
                - self.dic_balance_sheet["securities collateral"],
                0.0,
            )


        # Recursive calling of the missing amount of collateral through the ending of the own repos of the bank agent.
        self.end_repos(missing_collateral)

        # Recompute the missing amount of collateral after the end of the
        # recursive algorithm.
        if self.Network.substitution:
            #  We allow the bank self to use its securities usable as substitute for the securities she received in the first place
            missing_collateral = max(
                amount
                - self.dic_balance_sheet["securities collateral"] - self.dic_balance_sheet["securities usable"],
                0.0,
            )
        
        else:
            # otherwise only the securities collateral can be used
            missing_collateral = max(
                amount
                - self.dic_balance_sheet["securities collateral"],
                0.0,
            )

        # test if the recursive algorithm worked adequately
        assessment =  missing_collateral <= par.float_limit
        if not(assessment):
            self.Network.str_output_error = f"***ERROR***: bank {self.id} has not enough collateral to end its reverse repo with bank {bank_id}, missing amount is {missing_collateral}. Plot and stop simulation. \n" + self.__str__()
            print(self.Network.str_output_error)

        # Update all the required balance sheet items by the closing of the
        # reverse repo.
        if not(self.Network.beta_new):
            assessment = (
                self.dic_balance_sheet["securities reused"] > par.float_limit
                and self.dic_balance_sheet["securities usable"] > par.float_limit
            )
            
            if not(assessment):
                self.Network.str_output_error = f"***ERROR***: both reused {self.dic_balance_sheet['securities reused']} and usable {self.dic_balance_sheet['securities usable']} are positive, \n while normally supposed to use all usable before using collateral. Plot and stop simulation." + self.__str__()
                print(self.Network.str_output_error)


        assessment =  (
            abs(
                self.dic_balance_sheet["securities collateral"]
                + self.dic_balance_sheet["securities reused"]
                - self.dic_balance_sheet["reverse repo balance"]
            )
            < par.float_limit
        )
        if not(assessment):
            self.Network.str_output_error = f"***ERROR***: incorrect balance sheet \n securities collateral {self.dic_balance_sheet['securities collateral']}\n, securities reused {self.dic_balance_sheet['securities reused']}\n, reverse repos {self.dic_balance_sheet['reverse repo balance']} \n, difference {self.dic_balance_sheet['reverse repo balance'] - self.dic_balance_sheet['securities collateral'] - self.dic_balance_sheet['securities reused']}. Plot and stop simulation."
            print(self.Network.str_output_error)

        # Update of the balance sheet items
        self.dic_balance_sheet["cash"] += amount
        self.dic_balance_sheet["reverse repo balance"] -= amount
        self.rev_repo_exp[bank_id] -= amount

        if self.Network.substitution:

            # split between colat and usable
            amount_col_part = min(self.dic_balance_sheet["securities collateral"],amount)
            amount_usable_part = max(amount - amount_col_part,0.0)

            self.dic_balance_sheet["securities collateral"] -= amount_col_part

            # substitution
            self.dic_balance_sheet["securities usable"] -= amount_usable_part
            self.dic_balance_sheet["securities encumbered"] += amount_usable_part
            self.dic_balance_sheet["securities reused"] -= amount_usable_part

            ## ! WARNING ! it doesn't update the off_balance_repo exposure of self with an other bank...
        
        else:
            self.dic_balance_sheet["securities collateral"] -= amount

        # update df_reverse_repos
        # initialize the list of trans_ids with status == True
        trans_ids = self.df_rev_repo_trans.loc[bank_id][
            self.df_rev_repo_trans.loc[bank_id]["status"]
        ].index

        # case with an issue (no solution yet, just print the issue and a warning)
        # the issue came from the recursivity of the algo, solved due to the limited recusrsion 
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
                f"WARNING: df_reverse_repo and on_balance_repo/off_balance_repos do not match ! for lender {self.id} and borrower {bank_id} for the amount {amount}"
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

        # # Once a reverse repo is ended, there is an excess liquidity which
        # # requires closing the own repos of the bank self
        # self.step_end_repos()

    def enter_repos(self):
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

        # if there is no LCR mgt (no ECB funding to mgt LCR) or if we allow money creation, we might have a bigger shock to absorb than the available collateral, so only a part of the shock is absorded on the repo market
        if not (self.Network.LCR_mgt_opt) or self.Network.loan_tenor:
            repo_ask = min(
                repo_ask,
                self.dic_balance_sheet["securities usable"]
                * self.collateral_value
                + self.dic_balance_sheet["securities collateral"]
                * self.collateral_value,
            )

        # Case disjunction: nothing to do if the repo_ask is lower than the min_repo_trans_size
        if repo_ask <= self.Network.min_repo_trans_size:
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
            
            # # fill chains_rev_repo with reuse (need to check that the usable of the bank borrowing cash are not sufficient so that reuse will be done)
            # if self.dic_balance_sheet["securities usable"] < repo_ask - rest:
            #     self.Network.add_rev_repo_to_chains(b,self.id)

            # Test if the bank agent owns enough collateral to enter
            # into this repo
            assessment =  (
                self.dic_balance_sheet["securities usable"]
                * self.collateral_value
                + self.dic_balance_sheet["securities collateral"]
                * self.collateral_value
                - (repo_ask - rest)
                > -par.float_limit
            )
            if not(assessment):
                self.Network.str_output_error = f"***ERROR***: not Enough Collateral for bank {self.id}. Plot and stop simulation."
                print(self.Network.str_output_error)

            # first use the securities usable then the securities collateral
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

            if not(self.Network.beta_new): # this test is only valid in case of absnce of collateral creation which is added as usable
                assessment =  not (
                    self.dic_balance_sheet["securities reused"] > par.float_limit
                    and self.dic_balance_sheet["securities usable"]
                    > par.float_limit
                )
                
                if not(assessment):
                    self.Network.str_output_error = f"***ERROR***: both reused {self.dic_balance_sheet['securities reused']} and usable {self.dic_balance_sheet['securities reused']} are positive, while normally supposed to use all usable before using collat. Plot and stop simulation."
                    print(self.Network.str_output_error)
                

            repo_ask = rest
            if rest <= self.Network.min_repo_trans_size or len(bank_list) == 0:
                break

        # check for errors, in case of conservative shocks all repo request should be satisfied
        if self.Network.conservative_shock:
            assessment =  repo_ask <= par.float_limit
            if not(assessment):
                self.Network.str_output_error = f"***ERROR***: repo request unsatified for bank {self.id} for the amount {repo_ask}. Plot and stop simulation."
                print(self.Network.str_output_error)

    def enter_reverse_repo(self, bank_id, amount):
        """
        Instance method allowing the instance of the ClassBank to enter into
        a reverse repo with bank bank_id upon request of this latter.
        :param self: the bank bank_id requesting a repo.
        :param bank_id: the bank that requests the repo
        :param amount: the amount of cash requested.
        :return: the remaining amount of repo to the requested by the bank
        bank_id
        """

        # OPTION: A bank does not accept to enter into a reverse repo if he is already somewhere is the collateral chain (need to check that the usable of the bank borrowing cash are not sufficient so that reuse will be done)
        # if self.Network.will_create_a_loop(self.id,bank_id) and (self.banks[bank_id].dic_balance_sheet["securities usable"] < amount):
        #     return amount

        reverse_accept = max(
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"],
            0.0,
        )

        # Test if a bank is lending to itself due to a float error.
        assessment = self.id != bank_id
        if not(assessment):
            self.Network.str_output_error = f"***ERROR***: Bank {self.id} is lending to itself. Plot and stop simulation."
            print(self.Network.str_output_error)

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

    def step_enter_central_bank_funding(self):
        """
        In case shocks are non conversative banks may request cash to the central bank as a last resort when there is not enough cash available on the repo market.
        In case of LCR management, banks may request cash to the central bank as a last resort when there is not enough collateral available for performing repos.
        In case of money creation, banks may request cash to the central bank as a last resort when there is not enough collateral available for performing repos.
        """

        # Define the amount of CB funds to be requested
        cb_funding_ask = -(
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
        )

        # Case disjunction: nothing to do if the repo_ask is negative
        if cb_funding_ask <= 0.0:
            return

        else:
            # perform a central bank funding
            self.dic_balance_sheet["central bank funding"] += cb_funding_ask
            self.dic_balance_sheet["cash"] += cb_funding_ask

    def choose_bank(self, bank_list):
        trusts = {}
        for b in bank_list:
            trusts[b] = self.trust[b]
        return max(trusts, key=trusts.get)

    def update_learning(self, bank, value):
        # Lux's approach: creates a trust between 0 and 1, when value is between 0 and 1, deacing in power 2 toward 0 in case value is 0, converging in power 2 toward 1 in case value is 1
        if not(self.Network.learning_speed):
            self.trust[bank] = np.random.uniform(0,1)
        else:
            self.trust[bank] = self.trust[bank] + self.Network.learning_speed * (value - self.trust[bank])

    def close_maturing_loans(self):
        "It the exact opposite of money creation. Decrease loans and deposits in the same proportions."
        
        if self.Network.step in self.dic_loans_steps_closing:

            # no effect on cash due to the combined effect of the decrease from deposits and increase from loan payback
            self.dic_balance_sheet["loans"] -= self.dic_loans_steps_closing[self.Network.step]
            self.dic_balance_sheet["deposits"] -= self.dic_loans_steps_closing[self.Network.step]
    
    def close_maturing_securities(self):
        if self.Network.step in self.dic_loans_steps_closing:
            # close also maturing colat: we hope than the bank has sufficient collateral usable, otherwise we breach the equality reverse repo = collat recieved + collat reused, we also breach the asset = liability
            collat_decrease = self.Network.beta_new*self.dic_loans_steps_closing[self.Network.step]/(1-self.Network.beta_new)
            self.dic_balance_sheet["securities usable"] -= collat_decrease / self.collateral_value
            self.dic_balance_sheet["deposits"] -= collat_decrease

            assessment =  (
                self.dic_balance_sheet["securities usable"]
                < -par.float_limit
            )
            if not(assessment):
                self.Network.str_output_error = f"***ERROR***: not enough securities usable to allow the maturing of the securities. Plot and stop simulation."
                print(self.Network.str_output_error)
        
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
        an instance of ClassBank. In case of absence of counterparty credit risk the leverage exposure is exactly the total assets.
        Here we do not model the bilateral agreements allowing to net short and long exposure between counterparties exchanging repos in both directions.
        """
        leverage_exposure = (self.dic_balance_sheet["cash"]
            + self.dic_balance_sheet["securities usable"] * self.collateral_value
            + self.dic_balance_sheet["securities encumbered"] * self.collateral_value
            + self.dic_balance_sheet["loans"]
            + self.dic_balance_sheet["reverse repo balance"])
        return leverage_exposure

    def leverage_ratio(self):
        """
        Instance method computing the leverage ratio of an instance of
        ClassBank.
        :return:
        """
        return self.dic_balance_sheet["own funds"] / self.leverage_exposure()
        

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

        assessment = (
            self.dic_balance_sheet["cash"]
            - self.alpha * self.dic_balance_sheet["deposits"]
            >= -par.float_limit
        )
        if not (assessment):
            self.Network.str_output_error = f"***ERROR***: minimum reserves not respected for bank {self.id} at step {self.Network.step} \n The reverse repos provided to the rest of the network are {self.rev_repo_exp}. Plot and stop simulation."
            print(self.Network.str_output_error)

    def store_df_reverse_repos(self, path):
        self.df_rev_repo_trans.to_csv(f"{path}df_reverse_repos.csv")

    def total_assets(self):
        total_assets = 0
        for item in par.assets:
            total_assets += self.dic_balance_sheet[item]
        return total_assets
    
    def total_liabilities(self):
        total_liabilities = 0
        for item in par.liabilities:
            total_liabilities += self.dic_balance_sheet[item]
        return total_liabilities
    
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
        ----Central bank funding: {}\r
        ----------------------------------------\r
        Off-balance sheet Items:\r
        ----Collateral received: {}\r
        ----Collateral Reuse: {}\r
        ----------------------------------------\r
        Regulatory:\r
        ----Liquidity Coverage Ratio: {}%\r
        ----Cash / Deposits (for minimal reserves): {}%\r
        ----Leverage ratio: {}%\r
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

