import os
import numpy as np
from scipy.stats import pareto
from tqdm import tqdm
from bank import ClassBank
import pandas as pd
import parameters as par
from scipy import stats

class ClassNetwork:
    def __init__(
        self,
        nb_banks,
        initial_deposits_size,
        alpha_init,
        alpha,
        beta_init,
        beta_reg,
        beta_star,
        beta_new,
        gamma_init,
        gamma,
        gamma_star,
        gamma_new,
        collateral_value,
        initialization_method,
        alpha_pareto,
        shocks_method,
        shocks_law,
        shocks_vol,
        LCR_mgt_opt,
        min_repo_trans_size,
        loan_tenor,
        loan_period,
        new_loans_vol,
        new_loans_mean,
        end_repo_period,
        substitution,
        learning_speed,
        check_leverage_opt,
        init_money_min,
        QE_start,
        QE_stop,
    ):

        # adequacy tests
        assessment  = (initialization_method in par.initialization_methods)
        if not(assessment):
            self.str_output_error = "***ERROR***: invalid initialisation method. Plot and stop simulation."
            print(self.str_output_error)
        
        assessment  = (shocks_method in par.shocks_methods)
        if not(assessment):
            self.str_output_error = "***ERROR***: invalid shock method. Plot and stop simulation."
            print(self.str_output_error)
        
        # initialization of the class parameters.
        self.nb_banks = nb_banks
        self.initial_deposits_size = initial_deposits_size
        self.alpha_init = alpha_init
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_reg = beta_reg
        self.beta_star = beta_star
        self.beta_new = beta_new
        self.gamma_init = gamma_init
        self.gamma = gamma
        self.gamma_new = gamma_new
        self.gamma_star = gamma_star
        self.collateral_value = collateral_value
        self.initialization_method = initialization_method
        self.alpha_pareto = alpha_pareto
        self.shocks_method = shocks_method
        self.shocks_law = shocks_law
        self.shocks_vol = shocks_vol
        self.LCR_mgt_opt = LCR_mgt_opt
        self.min_repo_trans_size = min_repo_trans_size
        self.loan_tenor = loan_tenor
        self.loan_period = loan_period
        self.new_loans_vol = new_loans_vol
        self.new_loans_mean = new_loans_mean
        self.end_repo_period = end_repo_period
        self.substitution = substitution
        self.learning_speed = learning_speed
        self.check_leverage_opt = check_leverage_opt
        self.init_money_min = init_money_min
        self.QE_start = QE_start
        self.QE_stop = QE_stop

        # (Re)set the network
        self.reset_network()

    def reset_network(self):

        # time
        self.step = 0

        # instances of ClassBank in the Network
        self.banks = []

        # define the conservative shock paramater
        if self.shocks_method == "non-conservative":
            self.conservative_shock = False
        else:
            self.conservative_shock = True

        # initialize banks dataframe (exposures)
        self.df_banks = pd.DataFrame(
            index=range(self.nb_banks), columns=par.bank_items
        )

        # initialise the list of all the rev repo chains
        self.chains_rev_repo = []

        # initialise the chain of all the rev repo loops
        self.loops_rev_repo = []

        # Definition of the value of the collateral
        self.collateral_value = 1.0

        # initialize deposits in df_banks
        self.initialize_deposits()

        # initialize beta init to create inequal securities allocations
        self.initialize_beta_init()

        # build Banks
        for bank_id in range(self.nb_banks):
            self.banks.append(
                ClassBank(
                    Network=self,
                    id=bank_id,
                    initial_deposits=self.df_banks.loc[bank_id, "deposits"],
                    alpha_init=self.alpha_init,
                    alpha=self.alpha,
                    beta_init=self.ar_beta_init[bank_id],
                    beta_reg=self.beta_reg,
                    beta_star=self.beta_star,
                    gamma_init=self.gamma_init,
                    gamma=self.gamma,
                    collateral_value=self.collateral_value,
                )
            )

        # fill the recording data objects at step 0
        self.fill_step_df_banks()
        
        # initialize all banks
        for Bank in self.banks:
            Bank.initialize_banks(self.banks)

        # store initial loans
        self.df_banks["initial loans"] = self.df_banks["loans"]

        # initialize the output error string
        self.str_output_error = False

    def initialize_deposits(self):
        if self.initialization_method == "pareto":
            self.df_banks["deposits"] = (
                pareto.rvs(
                    self.alpha_pareto,
                    loc=0,
                    scale=1,
                    size=self.nb_banks,
                    random_state=None,
                )
                * self.initial_deposits_size
            )
        elif self.initialization_method == "constant":
            self.df_banks["deposits"] = np.ones(self.nb_banks) * self.initial_deposits_size

        # store initial deposits
        self.df_banks["initial deposits"] = self.df_banks["deposits"]


    def initialize_beta_init(self):
        if self.beta_init == "pareto":
            self.ar_beta_init = pareto.rvs(
                    self.alpha_pareto,
                    loc=0,
                    scale=0.1,
                    size=self.nb_banks,
                    random_state=None,
                ).clip(min=0,max=1)
        else:
            self.ar_beta_init = np.ones(self.nb_banks) * self.beta_init

    def step_network(self):

        # Defines an index of the banks
        index = np.arange(self.nb_banks)

        # generate new loansinitial_deposits_size
        if self.loan_tenor:
            ar_new_money = self.generate_new_money()

            # update the target for the mean reversion of the shocks
            self.df_banks["initial deposits"] += (1-self.gamma_new)*ar_new_money

        # generate shocks
        arr_shocks = self.generate_shocks()

        # loop 1: money creation + payment shocks + LCR mgt - enter or end cb funding
        index = np.random.permutation(index)  # permutation
        for bank_id in index:

            # create money
            if self.loan_tenor:
                if self.step % self.loan_period == 0:
                    self.banks[bank_id].set_money_creation(ar_new_money[bank_id])
                self.banks[bank_id].close_maturing_loans()
                if self.beta_new:
                    self.banks[bank_id].close_maturing_securities()

            # set the shocks (linear)
            self.banks[bank_id].set_shock(arr_shocks[bank_id])

            # LCR mgt (linear)
            if self.LCR_mgt_opt:
                self.banks[bank_id].lcr_mgt()

        # opt: put the loop 2 here
        # loop 2: leverage mgt - end existing repos
        index = np.random.permutation(index)  # permutation
        for bank_id in index:

            # opt 1: periodic end repo
            if self.end_repo_period: 
                if self.step % self.end_repo_period == 0:
                        self.banks[bank_id].step_end_repos()
            
            # opt 2: leverage mgt
            else: 
                self.banks[bank_id].leverage_mgt()

        # loop 3: reserve mgt - enter new repos
        index = np.random.permutation(index)  # permutation
        for bank_id in index:
            self.banks[bank_id].enter_repos()

            # in case of money creation only, last resorting refinancing is necessary (the only orthe way to borrow from CB is through LCR mgt that might not even work if there is an exces of collateral)
            if self.loan_tenor:
                self.banks[bank_id].step_enter_central_bank_funding()

            # opt: last resort refinancing for all cases  
            self.banks[bank_id].step_enter_central_bank_funding()

        # test with the loop 2 here 

        # # loop 4: fill df_banks and dic matrices
        self.fill_step_df_banks()

        # assert constraints (matricial)
        self.check_constraints()

        # new step of the network
        self.step += 1

    # not used 
    def will_create_a_loop(self,lender,borrower):
        output  = False
        for chain in self.chains_rev_repo:
            if lender in chain and borrower in chain:
                if chain.index(lender) > chain.index(borrower):
                    output = True
        return output

    # not used
    def add_rev_repo_to_chains(self,lender,borrower):
        "Adds a repo transaction to the list of all the existing chains in the network"
        
        # init
        new_chains = []
        lender_and_borrower_in_no_chain = True

        # loop on the chains 
        for chain in self.chains_rev_repo:

            # check if both lender and borroer are in the chain (risk of loop)
            if lender in chain and borrower in chain:
                lender_pos = chain.index(lender)
                borrower_pos = chain.index(borrower)

                # no loop if the new transaction is in the same direction
                if lender_pos < borrower_pos:
                    new_chain = chain[:lender_pos+1] + chain[borrower_pos:]
                    if new_chain not in new_chains:
                        new_chains.append(new_chain)
                
                # we have a loop otherwise
                elif lender_pos > borrower_pos:
                    new_loop = chain[borrower_pos:lender_pos+1]
                    if new_loop not in self.loops_rev_repo:
                        self.loops_rev_repo.append(new_loop)

            # complete the chain if the lender is at the end or the borrower at the begining of a given chain 
            elif lender==chain[-1]:
                lender_and_borrower_in_no_chain = False
                chain.append(borrower)
            elif borrower==chain[0]:
                lender_and_borrower_in_no_chain = False
                chain.insert(0,lender)

            # otherwise record the case where the lender or the borrwer (but not the two) are in a chain while avoiding duplicates 
            else: 
                if lender in chain:
                    lender_and_borrower_in_no_chain = False
                    pos = chain.index(lender)
                    new_chain = chain[:pos+1]
                    new_chain.append(borrower)
                    if new_chain not in new_chains:
                        new_chains.append(new_chain)

                if borrower in chain:
                    lender_and_borrower_in_no_chain = False
                    pos = chain.index(borrower)
                    new_chain = chain[pos:]
                    new_chain.insert(0,lender)
                    if new_chain not in new_chains:
                        new_chains.append(new_chain)


        # need to create the new chains previously recorded without duplicate 
        for new_chain in new_chains:
            self.chains_rev_repo.append(new_chain)

        # need to create a new chain if neither the borrower nor the lender are in any chain 
        if lender_and_borrower_in_no_chain:
            self.chains_rev_repo.append([lender,borrower])

    # not used
    def remove_rev_repo_from_chains(self, lender,borrower):
        
        # init
        new_chains_rev_repo = self.chains_rev_repo.copy()
        chains_with_lender = []
        chains_with_borrower = []
        downstream_chains = []
        upstream_chains = []

        # loop over all chains 
        for chain in self.chains_rev_repo:

            # look for chains with the pair sequence
            if lender in chain and borrower in chain:

                # if the two are in a chain they must be subsequent (otherwise a loop exists!)
                i = chain.index(lender)

                # build the two sub-chains
                downstream_chain = chain[:i+1]
                upstream_chain = chain[i+1:]

                # add them to the list of sub-chains 
                if len(downstream_chain)>1 and downstream_chain not in downstream_chains:
                    downstream_chains.append(downstream_chain)
                if len(upstream_chain)>1 and upstream_chain not in upstream_chains:
                    upstream_chains.append(upstream_chain)
                
                # remove this chain from the chains list 
                new_chains_rev_repo.remove(chain)
            
            # record all the chain in which the lender apears
            elif lender in chain and borrower not in chain:
                chains_with_lender.append(chain)
            
            # record all the chain in which the borrower apears
            if borrower in chain and lender not in chain:
                chains_with_borrower.append(chain)

        # add new chains if the lender (res. borrower) is not already part of a chain
        if not(chains_with_lender):
            new_chains_rev_repo += downstream_chains
        if not(chains_with_borrower):
            new_chains_rev_repo += upstream_chains

        # define the new list of chains 
        self.chains_rev_repo = new_chains_rev_repo 

    def check_constraints(self):
        self.check_balance_sheet()
        self.check_min_reserves()
        if self.LCR_mgt_opt:
            self.check_lcr()
        if not(self.end_repo_period) and self.check_leverage_opt:
            self.check_leverage()

    def check_balance_sheet(self):

        assessment = (
            (
                self.df_banks["total assets"]
                - self.df_banks["total liabilities"]
            )
            .abs()
            .lt(par.float_limit)
            .all()
        )

        if not(assessment):
            self.str_output_error = "***ERROR***: assets don't match liabilities for one or several banks. Plot and stop simulation."
            print(self.str_output_error)

    def check_min_reserves(self):

        assessment = (
            self.df_banks["excess liquidity"].gt(-par.float_limit).all()
        )

        if not (assessment):
            self.str_output_error = "***ERROR***: minimum reserves below its regulatory level for one or several banks. Plot and stop simulation."
            print(self.str_output_error)

    def check_lcr(self):

        assessment = (
            (
                self.df_banks["cash"]
                + self.df_banks["securities collateral"]
                * self.collateral_value
                + self.df_banks["securities usable"] * self.collateral_value
                - self.df_banks["deposits"] * self.beta_reg
            )
            .gt(-par.float_limit)
            .all()
        )

        if not (assessment):
            self.str_output_error = "***ERROR***: LCR not above its regulatory level for one or several banks. Plot and stop simulation."
            print(self.str_output_error)

    def check_leverage(self):

        assessment = (self.df_banks["own funds"] - self.df_banks["total assets"] * self.gamma).gt(-par.float_limit).all()

        if not (assessment):
            self.str_output_error = "***ERROR***: leverage ratio not above its regulatory level for one or several banks. Plot and stop simulation."
            print(self.str_output_error)

    def fill_step_df_banks(self):

        # -----------
        # accounting view

        # loop over the banks and accounting items
        for bank_id, Bank in enumerate(self.banks):
            for key in par.accounting_items:
                self.df_banks.loc[bank_id, key] = Bank.dic_balance_sheet[key]

        # matricial computation of constraints
        self.df_banks["total assets"] = self.df_banks[par.assets].sum(axis=1)
        self.df_banks["total liabilities"] = self.df_banks[
            par.liabilities
        ].sum(axis=1)
        self.df_banks["excess liquidity"] = (
            self.df_banks["cash"] - self.alpha * self.df_banks["deposits"]
        )
        
        if self.df_banks["deposits"].all() !=0:
            self.df_banks["reserve ratio"] = ((self.df_banks["cash"] / self.df_banks["deposits"]))*100
            self.df_banks["liquidity ratio"] = ((self.df_banks["cash"]
                    + self.df_banks["securities collateral"]
                    * self.collateral_value
                    + self.df_banks["securities usable"] * self.collateral_value) / (self.df_banks["deposits"]))*100
            self.df_banks["leverage ratio"] = (self.df_banks["own funds"] / self.df_banks["total assets"])*100
        else:
            self.df_banks["reserve ratio"] = 1
            self.df_banks["liquidity ratio"] = 1
            self.df_banks["leverage ratio"] = 1
        
        # others
        self.df_banks["borrowings"] = self.df_banks["repo balance"] + self.df_banks["central bank funding"]

    # not used ??
    def dump_step(self, path):
        self.df_banks.to_csv(f"{path}df_banks.csv")
        self.df_rev_repo_trans.to_csv(f"{path}df_reverse_repos.csv")

    def update_df_rev_repo_trans(self):

        # print
        print("get df_rev_repo_trans")

        dfs = []
        for Bank in tqdm(self.banks):
            dfs.append(Bank.df_rev_repo_trans)
        self.df_rev_repo_trans = pd.concat(
            dfs,
            keys=range(self.nb_banks),
            names=["owner_bank_id", "bank_id", "trans_id"],
            axis=0,
        )

    def generate_shocks(self):
        if self.shocks_method == "bilateral":
            arr_shocks = self.generate_bilateral_shocks()
        elif self.shocks_method == "multilateral":  # Damien's proposal,
            # doesn't work yet, could be enhanced
            arr_shocks = self.generate_multilateral_shocks()
        elif self.shocks_method == "dirichlet":
            arr_shocks = self.generate_dirichlet_shocks(
                option="mean-reverting"
            )
        elif self.shocks_method == "non-conservative":
            arr_shocks = self.generate_non_conservative_shocks()

        # Tests
        assessment = (np.min(self.df_banks["deposits"] + arr_shocks) >= 0)
        if not (assessment):
            self.str_output_error = "***ERROR***: negative shocks larger than deposits. Plot and stop simulation."
            print(self.str_output_error)

        if self.conservative_shock:
            assessment = (abs(arr_shocks.sum()) == 0.0)
            if not (assessment):
                self.str_output_error = f"***ERROR***: Shock doesn't sum to zero but to {arr_shocks.sum()}. Plot and stop simulation."
                print(self.str_output_error)

        return arr_shocks

    def generate_bilateral_shocks(self):
        # define middle of the list of banks
        N_max = (
            len(self.df_banks["deposits"]) - len(self.df_banks["deposits"]) % 2
        )  # can not apply a shock on
        # one bank if odd nb
        N_half = int(len(self.df_banks["deposits"]) / 2)

        # create a permutation of all the deposits amounts
        ix = np.arange(len(self.df_banks["deposits"]))  # create an index
        ix_p = np.random.permutation(ix)  # permutation of the index
        deposits_p = self.df_banks["deposits"][
            ix_p
        ]  # define the permuted array of deposits

        # apply a negative relative shock on the first half of the banks
        if self.shocks_law == "uniform":
            rho_1 = np.random.uniform(-1, 0, size=N_half)

        elif self.shocks_law == "beta":
            rho_1 = -np.random.beta(1, 1, size=N_half)

        elif self.shocks_law == "normal":
            norm_lower = -1
            norm_upper = 0
            mu = 0
            rho_1 = stats.truncnorm(
                (norm_lower - mu) / self.shocks_vol,
                (norm_upper - mu) / self.shocks_vol,
                loc=mu,
                scale=self.shocks_vol,
            ).rvs(N_half)

        # apply a positive relative shock on the second half of the banks
        rho_2 = -rho_1 * deposits_p[0:N_half] / deposits_p[N_half:N_max]

        # concatenate the relative shocks
        if len(self.df_banks["deposits"]) > N_max:
            rho = np.concatenate([rho_1, rho_2, [0]])
        elif len(self.df_banks["deposits"]) == N_max:
            rho = np.concatenate([rho_1, rho_2])

        # build an un-permuted array of absolute shocks
        shocks = np.zeros(len(self.df_banks["deposits"]))

        # compute the absolute shock from the deposit amount
        shocks[ix_p] = deposits_p * rho

        return shocks

    def generate_multilateral_shocks(self):
        # define middle of the list of banks
        N_max = (
            len(self.df_banks["deposits"]) - len(self.df_banks["deposits"]) % 2
        )  # can not apply a shock on
        # one bank if odd nb
        N_half = int(len(self.df_banks["deposits"]) / 2)

        # create a permutation of all the deposits amounts
        ix = np.arange(len(self.df_banks["deposits"]))  # create an index
        ix_p = np.random.permutation(ix)  # permutation of the index
        deposits_p = self.df_banks["deposits"][
            ix_p
        ]  # define the permuted array of deposits

        # apply a shock on the first half of the banks
        if self.shocks_law == "uniform":
            rho = np.random.uniform(-0.1, 0.1, size=N_max)  # case uniform  law

        elif self.shocks_law == "beta":
            rho = -np.random.beta(1, 1, size=N_half)  # case beta  law

        rho_1 = rho[0:N_half]
        rho_2 = rho[N_half:N_max]

        correction_factor = -(
            np.sum(rho_1 * deposits_p[0:N_half])
            / np.sum(rho_2 * deposits_p[N_half:N_max])
        )

        rho_2 = rho_2 * correction_factor

        # concatenate the relative shocks
        if len(self.df_banks["deposits"]) > N_max:
            rho = np.concatenate([rho_1, rho_2, [0]])
        elif len(self.df_banks["deposits"]) == N_max:
            rho = np.concatenate([rho_1, rho_2])

        # build an un-permuted array of absolute shocks
        shocks = np.zeros(len(self.df_banks["deposits"]))

        # compute the absolute shock from the deposit amount
        shocks[ix_p] = deposits_p * rho

        return shocks

    def generate_dirichlet_shocks(self, option):

        std_control = 1.0 / (self.shocks_vol**2.0)

        if option == "dynamic":
            dispatch = np.random.dirichlet(
                (
                    np.abs(self.df_banks["deposits"] + 1e-8)
                    / self.df_banks["deposits"].sum()
                )
                * std_control
            )
        elif option == "static":
            dispatch = np.random.dirichlet(
                (
                    np.ones(len(self.df_banks["deposits"]))
                    / len(self.df_banks["deposits"])
                )
                * std_control
            )
        elif option == "mean-reverting":
            dispatch = np.random.dirichlet(
                (
                    self.df_banks["initial deposits"]
                    / self.df_banks["initial deposits"].sum()
                )
                * std_control
            )

        new_deposits = self.df_banks["deposits"].sum() * dispatch
        shocks = new_deposits - self.df_banks["deposits"]

        return shocks

    def generate_non_conservative_shocks(self):
        if self.shocks_law == "log-normal":
            std_control = np.sqrt(np.log(1.0 + self.shocks_vol**2.0))
            new_deposits = (
                np.random.lognormal(
                    mean=-0.5 * std_control**2,
                    sigma=std_control,
                    size=self.nb_banks,
                )
                * self.df_banks["deposits"]
            )

        elif self.shocks_law == "normal":
            new_deposits = np.maximum(
                self.df_banks["deposits"]
                + np.random.randn(self.nb_banks)
                * self.shocks_vol,
                0.0,
            )

        elif (
            self.shocks_law == "normal-mean-reverting"
        ):  # lux approahc + a clip to the negative side to avoid withdrawing more deposits than initially existing
            mean_reversion = self.shocks_vol
            epsilon = np.random.normal(
                loc=0,
                scale=self.shocks_vol,
                size=len(self.df_banks["deposits"]),
            )
            shocks = (
                mean_reversion
                * (
                    self.df_banks["initial deposits"]
                    - self.df_banks["deposits"]
                )
                + epsilon * self.df_banks["deposits"]
            )

            # center the shocks
            shocks = shocks - np.mean(shocks)

            # clip the negative shocks to the deposits size
            new_deposits = (self.df_banks["deposits"] + shocks).clip(lower=0)

        shocks = new_deposits - self.df_banks["deposits"]
        return shocks
    

    def generate_new_money(self):
                
        # case if no initialization
        if not(self.initial_deposits_size):
            
            if self.step == 0:
                ar_current_money = np.ones(self.nb_banks)*self.init_money_min
            else:
                # rich get richer
                std_control = np.sqrt(np.log(1.0 + self.new_loans_vol**2))
                ar_random = (np.random.lognormal(mean=-0.5 * std_control**2,sigma=std_control,size=self.nb_banks))
                ar_new_money = ar_random*self.new_loans_mean*ar_current_money
        

        # case with initialization
        else: 
            initial_money = self.df_banks["initial loans"] / (1-self.beta_new*(1-self.gamma_new))
            
            # no vol around the growth
            ar_new_money = self.new_loans_mean*initial_money
            
            # opt: gaussian vol around the growth
            # ar_new_money = (np.random.randn(self.nb_banks) * self.new_loans_vol + 1)*self.new_loans_mean*initial_money
            
        return ar_new_money
