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

        # initialize the matrices dictionary (exposures)
        self.dic_matrices = dict.fromkeys(
            par.matrices, np.zeros((self.nb_banks, self.nb_banks))
        )

        # Definition of the value of the collateral
        self.collateral_value = 1.0

        # build deposits and create
        for bank_id in range(self.nb_banks):
            if self.initialization_method == "pareto":
                self.df_banks.loc[bank_id, "deposits"] = (
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
                self.df_banks.loc[bank_id, "deposits"] = 100.0
            self.df_banks.loc[bank_id, "initial deposits"] = self.df_banks.loc[
                bank_id, "deposits"
            ]
            self.banks.append(
                ClassBank(
                    Network=self,
                    id=bank_id,
                    initial_deposits=self.df_banks.loc[bank_id, "deposits"],
                    alpha_init=self.alpha_init,
                    alpha=self.alpha,
                    beta_init=self.beta_init,
                    beta_reg=self.beta_reg,
                    beta_star=self.beta_star,
                    gamma=self.gamma,
                    nb_banks=self.nb_banks,
                    collateral_value=self.collateral_value,
                    conservative_shock=self.conservative_shock,
                    LCR_mgt_opt=self.LCR_mgt_opt,
                )
            )

        # initialize all banks
        for Bank in self.banks:
            Bank.initialize_banks(self.banks)

        # fill the recording data objects at step 0
        self.step_fill()

    def step_network(self):
        """
        Instance method allowing the computation of the next step status of
        the network.
        :return:
        """

        # Generation of the shocks
        if self.shocks_method == "bilateral":
            shocks = sh.generate_bilateral_shocks(
                self.df_banks["deposits"],
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "multilateral":  # Damien's proposal,
            # doesn't work yet, could be enhanced
            shocks = sh.generate_multilateral_shocks(
                self.df_banks["deposits"],
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "dirichlet":
            shocks = sh.generate_dirichlet_shocks(
                self.df_banks["deposits"],
                self.df_banks["initial deposits"],
                option="mean-reverting",
                vol=self.shocks_vol,
            )
        elif self.shocks_method == "non-conservative":
            shocks = sh.generate_non_conservative_shocks(
                self.df_banks["deposits"],
                self.df_banks["initial deposits"],
                self.df_banks["total assets"],
                law=self.shocks_law,
                vol=self.shocks_vol,
            )
        else:
            assert False, ""

        # Tests to ensure the shock created matches the required properties
        assert (
            np.min(self.df_banks["deposits"] + shocks) >= 0
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
        index = np.arange(self.nb_banks)  # Defines an index of the banks
        for bank_id in index:
            self.banks[bank_id].set_shock(shocks[bank_id])
            self.banks[bank_id].set_collateral(self.collateral_value)
            if self.LCR_mgt_opt:
                self.banks[bank_id].step_lcr_mgt()

        # Permutation of the banks' indexes to decide in which order banks can close their repos.
        index = np.random.permutation(index)
        for bank_id in index:
            self.banks[bank_id].step_end_repos()

        # New permutation of the banks' indexes to decide in which order banks can enter into repos
        index = np.random.permutation(index)
        for bank_id in index:
            self.banks[bank_id].step_enter_repos()
            if not (self.conservative_shock) or not (self.LCR_mgt_opt):
                self.banks[bank_id].step_central_bank_funding()

        # loop 4: assert constraints and fill data
        for bank_id in index:
            self.banks[bank_id].assert_minimum_reserves()
            self.banks[bank_id].assert_alm()
            if self.LCR_mgt_opt:
                self.banks[bank_id].assert_lcr()
            self.step_fill_single_bank(bank_id)
        self.step_fill()

        # new step of the network
        self.step += 1

    def step_fill_single_bank(self, bank_id):

        Bank = self.banks[bank_id]
        df = Bank.df_reverse_repos

        # fill df_banks
        for item in par.accounting_items:
            if item in par.assets:
                self.df_banks.loc[bank_id, item] = Bank.assets[item]
            elif item in par.liabilities:
                self.df_banks.loc[bank_id, item] = Bank.liabilities[item]
            elif item in par.off_bs_items:
                self.df_banks.loc[bank_id, item] = Bank.off_bs_items[item]

        # fill dic_matrices
        self.dic_matrices["adjency"][bank_id, :] = np.array(
            list(self.banks[bank_id].reverse_repos.values())
        )
        trusts = list(self.banks[bank_id].trust.values())  # nb_banks-1 items
        self.dic_matrices["trust"][bank_id, :bank_id] = trusts[:bank_id]
        self.dic_matrices["trust"][bank_id, bank_id + 1 :] = trusts[bank_id:]

    def step_fill(self):
        # fill df_banks
        self.df_banks["total assets"] = self.df_banks[par.assets].sum(axis=1)
        self.df_banks["excess liquidity"] = (
            self.df_banks["cash"] - self.alpha * self.df_banks["deposits"]
        )

        # fill dic_matrices
        self.dic_matrices["adjency"][self.dic_matrices["adjency"] < 0] = 0
        self.dic_matrices["non-zero_adjency"] = self.dic_matrices["adjency"][
            np.nonzero(self.dic_matrices["adjency"])
        ]
        self.dic_matrices["binary_adjency"] = np.where(
            self.dic_matrices["adjency"] > self.min_repo_size, True, False
        )

    def store_network(self, path):
        self.df_banks.to_csv(f"{path}df_banks.csv")
        self.df_reverse_repos.to_csv(f"{path}df_reverse_repos.csv")

    def build_df_reverse_repos(self):
        dfs = []
        for Bank in self.banks:
            dfs.append(Bank.df_reverse_repos)
        self.df_reverse_repos = pd.concat(
            dfs,
            keys=range(self.nb_banks),
            names=["owner_bank_id", "bank_id", "trans_id"],
            axis=0,
        )
