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
import pandas as pd
import parameters as par


class ClassNetwork:
    def __init__(
        self,
        nb_banks,
        alpha_init,
        alpha,
        beta_init,
        beta_reg,
        beta_star,
        gamma,
        collateral_value,
        initialization_method,
        alpha_pareto,
        shocks_method,
        shocks_law,
        shocks_vol,
        min_repo_size,
        LCR_mgt_opt=True,
    ):

        # adequacy tests
        assert (
            initialization_method in par.initialization_methods
        ), "Invalid initialisation method"
        assert shocks_method in par.shocks_methods, "Invalid shock method"

        # initialization of the class parameters.
        self.nb_banks = nb_banks
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
        self.shocks_law = shocks_law
        self.shocks_vol = shocks_vol
        self.min_repo_size = min_repo_size
        self.LCR_mgt_opt = LCR_mgt_opt

        # (Re)set the network
        self.reset_network()

    def reset_network(self):

        self.step = 0  # Step number in the simulation process
        self.banks = []  # instances of ClassBank in the Network

        if self.shocks_method == "non-conservative":
            self.conservative_shock = False
        else:
            self.conservative_shock = True

        # initialize banks dataframe
        self.df_banks = pd.DataFrame(
            index=range(self.Network.nb_banks), columns=par.accounting_items
        )

        # Definition of the value of the collateral
        self.collateral_value = 1.0

        # build deposits and create
        for b in range(self.nb_banks):
            if self.initialization_method == "pareto":
                self.df_banks.loc[b, "Deposits"] = (
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
                self.df_banks.loc[b, "Deposits"] = 100.0
            self.df_banks.loc[b, "Initial deposits"] = self.df_banks.loc[
                b, "Deposits"
            ]
            self.banks.append(
                ClassBank(
                    id=b,
                    initial_deposits=self.df_banks.loc[b, "Deposits"],
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

        # initialize all banks
        for Bank in self.banks:
            Bank.initialize_banks(self.banks)

    def step_network(self):
        """
        Instance method allowing the computation of the next step status of
        the network.
        :return:
        """

        # Generation of the shocks
        if self.shocks_method == "bilateral":
            shocks = sh.generate_bilateral_shocks(
                self.df_banks["Deposits"],
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "multilateral":  # Damien's proposal,
            # doesn't work yet, could be enhanced
            shocks = sh.generate_multilateral_shocks(
                self.df_banks["Deposits"],
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "dirichlet":
            shocks = sh.generate_dirichlet_shocks(
                self.df_banks["Deposits"],
                self.df_banks["Initial deposits"],
                option="mean-reverting",
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "non-conservative":
            shocks = sh.generate_non_conservative_shocks(
                self.df_banks["Deposits"],
                self.df_banks["Initial deposits"],
                self.df_banks["Total assets"],
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        else:
            assert False, ""

        # Tests to ensure the shock created matches the required properties
        assert (
            np.min(self.df_banks["Deposits"] + shocks) >= 0
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
        ix = np.arange(self.nb_banks)  # Defines an index of the banks
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

        # now we are at a new step of the network !
        self.step += 1

    def fill_df_banks(self):
        for i, Bank in enumerate(self.banks):
            for accounting_item in par.accounting_items:
                self.df_banks.loc[i,accounting_item] = 

        return 0
