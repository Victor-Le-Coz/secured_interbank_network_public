import os

# os.environ["OMP_NUM_THREADS"] = "1"

import networkx as nx
import numpy as np
from tqdm import tqdm
import graphics as gx
import functions as fct
from network import ClassNetwork
import pandas as pd
import parameters as par
import emp_preprocessing as ep
import emp_metrics as em
from graphics import ClassGraphics


class ClassDynamics:
    def __init__(
        self,
        Network,
        nb_steps,
        path_results,
        dump_period,
        plot_period,
        cp_option,
    ):
        # initialization of the class parameters.
        self.Network = Network
        self.nb_steps = nb_steps
        self.path_results = path_results
        self.dump_period = dump_period
        self.cp_option = cp_option

        # Create the required path to store the results
        fct.delete_n_init_path(self.path_results)

        # network trajectories
        self.df_network_trajectory = pd.DataFrame(index=range(self.nb_steps))

        # individual trajectory
        self.single_bank_id = 0  # the selected single bank id
        self.df_bank_trajectory = pd.DataFrame(index=range(self.nb_steps))

        # total assets of each bank
        self.arr_total_assets = np.zeros(
            (self.nb_steps, self.Network.nb_banks), dtype=np.float32
        )

        # reverse repo adj exposure: array (nb_steps * nb_banks * nb_banks)
        self.arr_rev_repo_exp_adj = np.zeros(
            (self.nb_steps, self.Network.nb_banks, self.Network.nb_banks),
            dtype=np.float32,
        )

        # binary adj exposure: dic of array of binary adj matrices
        self.dic_arr_binary_adj = {}
        for agg_period in par.agg_periods:
            self.dic_arr_binary_adj.update(
                {
                    agg_period: np.zeros(
                        (
                            self.nb_steps,
                            self.Network.nb_banks,
                            self.Network.nb_banks,
                        )
                    )
                }
            )

        # initialise the class graphics
        self.Graphics = ClassGraphics(Dynamics=self, plot_period=plot_period)

    def fill_step(self):

        # -----------
        # accounting view

        self.fill_step_df_network_trajectory()
        self.fill_step_df_bank_trajectory()
        self.fill_step_arr_total_assets()

    def fill_step_df_network_trajectory(self):

        # -----------
        # accounting view

        # accounting items
        for item in par.bank_items:
            self.df_network_trajectory.loc[
                self.Network.step, f"{item} tot. network"
            ] = self.Network.df_banks[item].sum()

        # gini
        self.df_network_trajectory.loc[self.Network.step, "gini"] = em.gini(
            self.Network.df_banks["total assets"]
        )

        # Collateral reuse
        self.df_network_trajectory.loc[
            self.Network.step, "collateral reuse"
        ] = (
            self.df_network_trajectory.loc[
                self.Network.step, "securities reused tot. network"
            ]
        ) / (
            self.df_network_trajectory.loc[
                self.Network.step, "securities collateral tot. network"
            ]
            + 1e-10
        )

    def fill_step_df_bank_trajectory(self):

        # -----------
        # accounting view

        # Build the time series of the accounting items
        for item in par.bank_items:
            self.df_bank_trajectory.loc[
                self.Network.step, item
            ] = self.Network.df_banks.loc[self.single_bank_id, item]

    def fill_step_arr_total_assets(self):
        self.arr_total_assets[self.Network.step] = np.array(
            self.Network.df_banks["total assets"]
        )

    def fill(self):

        # --------------------------------
        # exposure view & transaction view

        # build df_rev_repo_trans bank level data
        self.Network.get_df_rev_repo_trans()

        # build adj matrices from df_rev_repo_trans
        self.build_adj_matrices()

        # fill df_network_trajectory & df_bank_trajectory from the adj matrices
        self.fill_dic_degree()
        self.fill_df_network_trajectory()
        self.fill_df_bank_trajectory()

        # save the data frame results
        self.df_bank_trajectory.to_csv(
            f"{self.path_results}df_bank_trajectory.csv"
        )
        self.df_network_trajectory.to_csv(
            f"{self.path_results}df_network_trajectory.csv"
        )
        self.Network.dump_step(self.path_results)

    def fill_df_network_trajectory(self):

        # --------------
        # exposure view

        # expost jaccard
        df_jaccard = em.get_rev_repo_exposure_stats(
            self.dic_arr_binary_adj, range(self.Network.step)
        )
        self.df_network_trajectory[
            [f"jaccard index-{agg_period}" for agg_period in par.agg_periods]
        ] = df_jaccard

        # expost density
        df_density = em.get_density(
            self.dic_arr_binary_adj, range(self.Network.step)
        )
        self.df_network_trajectory[
            [f"network density-{agg_period}" for agg_period in par.agg_periods]
        ] = df_density

        # expost av. degree
        df_degree = em.get_av_degree(
            self.dic_degree,
            range(self.Network.step),
        )
        self.df_network_trajectory[
            [f"av. degree-{agg_period}" for agg_period in par.agg_periods]
        ] = df_degree

        # expost repo exposures stats
        df_exposures_stats = em.get_exposure_stats(
            self.arr_rev_repo_exp_adj,
            range(self.Network.step + 1),
        )
        self.df_network_trajectory[
            [
                "repo exposures min network",
                "repo exposures max network",
                "repo exposures av. network",
            ]
        ] = df_exposures_stats

        # --------------
        # transaction view

        # expost reverse repo transactions stats network level
        name = "av. network"
        cols = [
            f"repo transactions maturity {name}",
            f"repo transactions notional {name}",
            f"number repo transactions {name}",
        ]
        df_transaction_stats = em.get_transaction_stats(
            df_trans=self.Network.df_rev_repo_trans,
            name=name,
            days=range(self.Network.step + 1),
        )
        self.df_network_trajectory[cols] = df_transaction_stats

    def fill_df_bank_trajectory(self):

        # --------------
        # exposure view

        # expost in-degree
        df_in_degree = pd.DataFrame()
        for agg_period in par.agg_periods:
            df_in_degree[f"av. in-degree-{agg_period}"] = self.dic_in_degree[
                agg_period
            ][:, self.single_bank_id]
        self.df_bank_trajectory[
            [f"av. in-degree-{agg_period}" for agg_period in par.agg_periods]
        ] = df_in_degree

        # expost out-degree
        df_out_degree = pd.DataFrame()
        for agg_period in par.agg_periods:
            df_out_degree[
                f"av. out-degree-{agg_period}"
            ] = self.dic_out_degree[agg_period][:, self.single_bank_id]
        self.df_bank_trajectory[
            [f"av. out-degree-{agg_period}" for agg_period in par.agg_periods]
        ] = df_out_degree

        # --------------
        # transaction view

        # expost reverse repo transactions stats bank level
        name = "av. bank"
        cols = [
            f"repo transactions maturity {name}",
            f"repo transactions notional {name}",
            f"number repo transactions {name}",
        ]
        df_trans = self.Network.df_rev_repo_trans

        # check of the single bank entered into any reverse repos
        if self.single_bank_id in df_trans.index.get_level_values(0):

            # filter the df on the single bank:
            df_trans = df_trans.loc[self.single_bank_id]

            # get transaction stats
            df_transaction_stats = em.get_transaction_stats(
                df_trans=df_trans,
                name=name,
                days=range(self.Network.step + 1),
            )

        # otherwise empty dataframe
        else:
            df_transaction_stats = pd.DataFrame(
                index=range(self.Network.step + 1), columns=cols
            )

        self.df_bank_trajectory[cols] = df_transaction_stats

    def fill_dic_degree(self):

        # --------------
        # exposure view

        # degree distribution
        (
            self.dic_in_degree,
            self.dic_out_degree,
            self.dic_degree,
        ) = em.get_degree_distribution(
            self.dic_arr_binary_adj,
            path=f"{self.path_results}exposure_view/degree_distribution/",
        )

    def get_arr_rev_repo_exp_adj(self):

        # ------------
        # exposure view

        # print
        print("get arr_rev_repo_exp_adj")

        # loop over the rows of df_reverse_repos transactions
        for index, row in tqdm(self.Network.df_rev_repo_trans.iterrows()):

            # get the tenor (fill by current step if tenor is empty)
            if np.isnan(row["tenor"]):
                tenor = self.Network.step - row["start_step"]
            else:
                tenor = row["tenor"]

            # add the amont of the transaction over its tenor
            for step in range(
                row["start_step"], row["start_step"] + tenor + 1
            ):
                self.arr_rev_repo_exp_adj[step, index[0], index[1]] += row[
                    "amount"
                ]

        # loop over the steps to clean values close to zero (or negative)
        for step in range(self.Network.step + 1):
            self.arr_rev_repo_exp_adj[step][
                self.arr_rev_repo_exp_adj[step] < 0
            ] = 0

        # save last step to csv
        fct.init_path(f"{self.path_results}exposure_view/adj_matrices/")
        fct.dump_np_array(
            self.arr_rev_repo_exp_adj[self.Network.step],
            f"{self.path_results}exposure_view/adj_matrices/arr_reverse_repo_adj_{self.Network.step}.csv",
        )

    def get_arr_binary_adj(self):

        # ------------
        # exposure view

        # print
        print("get arr_binary_adj (numba)")

        # convert list to array
        arr_agg_period = np.array(par.agg_periods)

        # build arr of results with numba
        arr_binary_adj = ep.fast_build_arr_binary_adj(
            self.arr_rev_repo_exp_adj, arr_agg_period, self.Network.step
        )

        # loop over agg periods
        for period_nb, agg_period in enumerate(par.agg_periods):

            # convert array results to dictionaries
            self.dic_arr_binary_adj[agg_period] = arr_binary_adj[period_nb]

            # save last step to csv
            fct.init_path(
                f"{self.path_results}exposure_view/adj_matrices/{agg_period}/"
            )
            fct.dump_np_array(
                arr_binary_adj[period_nb][self.Network.step],
                f"{self.path_results}exposure_view/adj_matrices/{agg_period}/arr_binary_adj_on_day_{self.Network.step}.csv",
            )

    def build_adj_matrices(self):

        # ------------
        # exposure view

        self.get_arr_rev_repo_exp_adj()
        self.get_arr_binary_adj()

    def simulate(self, output_keys=False):

        # record and store trajectories & parameters used at step 0
        self.save_param()
        self.fill_step()

        print("simulate the repo market")

        # simulate the network
        for _ in tqdm(range(self.nb_steps - 1)):

            # run one step of the network
            self.Network.step_network()

            # record information
            self.fill_step()

            # expoxt record, dump, and plot
            if self.Network.step % self.dump_period == 0:
                self.fill()
                self.Graphics.plot_all_trajectories()

        # store the final step (if not already done)
        if self.Network.step % self.dump_period != 0:
            self.fill()
            self.Graphics.plot_all_trajectories()

        # final print
        self.Graphics.plot_final_step()

    def save_param(self):
        with open(f"{self.path_results}input_parameters.txt", "w") as f:
            f.write(
                f"nb_banks={self.Network.nb_banks} \n"
                f"alpha={self.Network.alpha} \n"
                f"beta_init={self.Network.beta_init} \n"
                f"beta_reg={self.Network.beta_reg} \n"
                f"beta_star={self.Network.beta_star} \n"
                f"gamma={self.Network.gamma} \n"
                f"initialization_method={self.Network.initialization_method} \n"
                f"alpha_pareto={self.Network.alpha_pareto} \n"
                f"shock_method={self.Network.shocks_method} \n"
                f"shocks_law={self.Network.shocks_law} \n"
                f"shocks_vol={self.Network.shocks_vol} \n"
                f"min_repo_trans_size={self.Network.min_repo_trans_size} \n"
                f"nb_steps={self.nb_steps} \n"
                f"LCR_mgt_opt={self.Network.LCR_mgt_opt} \n"
            )


def single_run(
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
    result_location,
    min_repo_trans_size,
    nb_steps,
    dump_period,
    plot_period,
    cp_option,
    LCR_mgt_opt,
):

    # initialize ClassNetwork
    Network = ClassNetwork(
        nb_banks=nb_banks,
        alpha_init=alpha_init,
        beta_init=beta_init,
        beta_reg=beta_reg,
        beta_star=beta_star,
        alpha=alpha,
        gamma=gamma,
        collateral_value=collateral_value,
        initialization_method=initialization_method,
        alpha_pareto=alpha_pareto,
        shocks_method=shocks_method,
        shocks_law=shocks_law,
        shocks_vol=shocks_vol,
        min_repo_trans_size=min_repo_trans_size,
        LCR_mgt_opt=LCR_mgt_opt,
    )

    # initialize ClassDynamics
    dynamics = ClassDynamics(
        Network,
        nb_steps=nb_steps,
        path_results=result_location,
        dump_period=dump_period,
        plot_period=plot_period,
        cp_option=cp_option,
    )

    # simulate
    dynamics.simulate()
