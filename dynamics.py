import os
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


class ClassDynamics:
    def __init__(
        self,
        Network,
        nb_steps,
        path_results,
        dump_period,
        plot_period,
        cp_option,
        heavy_plot,
    ):
        # initialization of the class parameters.
        self.Network = Network
        self.nb_steps = nb_steps
        self.path_results = path_results
        self.dump_period = dump_period
        self.cp_option = cp_option
        
        self.plot_period = plot_period
        self.heavy_plot = heavy_plot

        # Create the required path to store the results
        fct.delete_n_init_path(self.path_results)

        # network trajectories
        self.df_network_trajectory = pd.DataFrame(index=range(self.nb_steps))

        # individual trajectory
        self.single_bank_id = 0  # the selected single bank id
        self.df_bank_trajectory = pd.DataFrame(index=range(self.nb_steps))

        # dashed trajectory for all banks
        self.dic_dashed_trajectory = {}

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

    def fill_step(self):

        # -----------
        # accounting view

        self.fill_step_df_network_trajectory()
        self.fill_step_df_bank_trajectory()

        if (
            self.Network.step % self.plot_period == 0
            or self.Network.step == self.nb_steps - 1
        ):
            self.fill_step_dic_dashed_trajectory()

    def fill_step_df_network_trajectory(self):

        # -----------
        # I - accounting view

        # accounting items
        for item in par.bank_items:
            self.df_network_trajectory.loc[
                self.Network.step, f"{item} tot. network"
            ] = self.Network.df_banks[item].sum()

        # gini
        self.df_network_trajectory.loc[
            self.Network.step, "gini"
        ] = em.fig_gini(self.Network.df_banks["total assets"])

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

    def fill_step_dic_dashed_trajectory(self):
        self.dic_dashed_trajectory.update(
            {self.Network.step: self.Network.df_banks}
        )

    def fill(self):

        # --------------------------------
        # exposure view & transaction view

        # build df_rev_repo_trans bank level data
        self.Network.update_df_rev_repo_trans()

        # build adj matrices from df_rev_repo_trans
        self.build_adj_matrices()

        # fill df_network_trajectory & df_bank_trajectory from the adj matrices
        self.fill_dic_degree()
        self.fill_df_network_trajectory()
        self.fill_df_bank_trajectory()

    def dump(self):

        # save the data frame results
        self.df_bank_trajectory.to_csv(
            f"{self.path_results}df_bank_trajectory.csv"
        )
        self.df_network_trajectory.to_csv(
            f"{self.path_results}df_network_trajectory.csv"
        )
        self.Network.df_rev_repo_trans.to_csv(
            f"{self.path_results}df_rev_repo_trans.csv"
        )
        os.makedirs(f"{self.path_results}/dashed_trajectory/", exist_ok=True)
        for step in self.dic_dashed_trajectory.keys():
            self.dic_dashed_trajectory[step].to_csv(
                f"{self.path_results}/dashed_trajectory/df_banks_on_day_{step}.csv"
            )

    def fill_df_network_trajectory(self):

        # --------------
        # II - exposure view

        # A - full path
        # expost jaccard
        df_jaccard = em.get_rev_repo_exposure_stats(
            self.dic_arr_binary_adj, range(self.Network.step)
        )
        cols = df_jaccard.columns
        self.df_network_trajectory[cols] = df_jaccard

        # expost density
        df_density = em.get_density(
            self.dic_arr_binary_adj, range(self.Network.step)
        )
        cols = df_density.columns
        self.df_network_trajectory[cols] = df_density

        # expost degree stats
        df_degree_stats = em.get_degree_stats(
            self.dic_degree,
            range(self.Network.step),
        )
        cols = df_degree_stats.columns
        self.df_network_trajectory[cols] = df_degree_stats

        # expost repo exposures stats
        df_exposures_stats = em.get_exposure_stats(
            self.arr_rev_repo_exp_adj,
            range(self.Network.step + 1),
        )
        cols = df_exposures_stats.columns
        self.df_network_trajectory[cols] = df_exposures_stats

        # B - dashed path
        days = range(self.Network.step + 1)  # to cover all steps up to now
        # expost cpnet
        if self.cp_option:
            df_cpnet = em.get_cpnet(
                dic_arr_binary_adj=self.dic_arr_binary_adj,
                arr_rev_repo_exp_adj=self.arr_rev_repo_exp_adj,
                days=days,
                plot_period=self.plot_period,
                path=f"{self.path_results}exposure_view/core-periphery/",
            )
            if self.heavy_plot:
                cols = df_cpnet.columns
            else:  # not save all columns if there is no need (spare memory)
                cols = [plot_char[0] for plot_char in par.cpnet_pvalue]
            self.df_network_trajectory[cols] = None
            for step in df_cpnet.index:
                self.df_network_trajectory.loc[step, cols] = df_cpnet.loc[step]

        # expost powerlaw
        if self.heavy_plot:
            df_powerlaw = em.get_powerlaw(
                dic_dashed_trajectory=self.dic_dashed_trajectory,
                days=days,
                plot_period=self.plot_period,
            )
            cols = df_powerlaw.columns
        else:  # not save all columns if there is no need (spare memory)
            cols = [plot_char[0] for plot_char in par.powerlaw_pvalue] + [
                plot_char[0] for plot_char in par.powerlaw_alpha
            ] + [plot_char[0] for plot_char in par.powerlaw_direction]
        
        self.df_network_trajectory[cols] = None
        if self.heavy_plot:
            for step in df_powerlaw.index:
                self.df_network_trajectory.loc[step, cols] = df_powerlaw.loc[step]

        # --------------
        # III - transaction view

        # expost reverse repo transactions stats network level
        df_transaction_stats = em.get_transaction_stats(
            df_rev_repo_trans=self.Network.df_rev_repo_trans,
            extension=" av. network",
            days=range(self.Network.step + 1),
        )
        cols = df_transaction_stats.columns
        self.df_network_trajectory[cols] = df_transaction_stats

    def fill_df_bank_trajectory(self):

        # --------------
        # exposure view

        # expost in-degree
        df_in_degree = pd.DataFrame()
        for agg_period in par.agg_periods:
            df_in_degree[f"in-degree-{agg_period}"] = self.dic_in_degree[
                agg_period
            ][:, self.single_bank_id]
        cols = df_in_degree.columns
        self.df_bank_trajectory[cols] = df_in_degree

        # expost out-degree
        df_out_degree = pd.DataFrame()
        for agg_period in par.agg_periods:
            df_out_degree[f"out-degree-{agg_period}"] = self.dic_out_degree[
                agg_period
            ][:, self.single_bank_id]
        cols = df_out_degree.columns
        self.df_bank_trajectory[cols] = df_out_degree

        # --------------
        # transaction view

        # expost reverse repo transactions stats bank level
        extension = " av. bank"
        df_trans = self.Network.df_rev_repo_trans

        # check of the single bank entered into any reverse repos
        if self.single_bank_id in df_trans.index.get_level_values(0):

            # filter the df on the single bank:
            df_trans = df_trans.loc[self.single_bank_id]

            # get transaction stats
            df_transaction_stats = em.get_transaction_stats(
                df_rev_repo_trans=df_trans,
                extension=" av. bank",
                days=range(self.Network.step + 1),
            )
            cols = df_transaction_stats.columns

        # otherwise empty dataframe
        else:
            cols = [
                f"repo transactions maturity{extension}",
                f"repo transactions notional{extension}",
                f"number repo transactions{extension}",
            ]
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

        # loop over the rows of df_rev_repo_trans
        for index, row in tqdm(self.Network.df_rev_repo_trans.iterrows()):

            # get the tenor (fill by current step if tenor is empty)
            if np.isnan(row["end_step"]):
                tenor = self.Network.step - row["start_step"]
            else:
                tenor = row["end_step"] - row["start_step"]

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
                self.arr_rev_repo_exp_adj[step] < self.Network.min_repo_trans_size
            ] = 0

        # save to csv for the plot_steps
        plot_steps = fct.get_plot_steps_from_period(
            range(self.Network.step + 1), self.plot_period
        )
        for step in plot_steps:
            os.makedirs(
                f"{self.path_results}exposure_view/adj_matrices/weighted/",
                exist_ok=True,
            )
            fct.dump_np_array(
                self.arr_rev_repo_exp_adj[self.Network.step],
                f"{self.path_results}exposure_view/adj_matrices/weighted/arr_reverse_repo_adj_{step}.csv",
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
            self.arr_rev_repo_exp_adj,
            arr_agg_period,
            self.Network.step,
            self.Network.min_repo_trans_size,
        )

        plot_steps = fct.get_plot_steps_from_period(
            range(self.Network.step + 1), self.plot_period
        )

        # loop over agg periods
        for period_nb, agg_period in enumerate(par.agg_periods):

            # convert array results to dictionaries
            self.dic_arr_binary_adj[agg_period] = arr_binary_adj[period_nb]

            # save to csv for the plot_steps
            for step in plot_steps:
                os.makedirs(
                    f"{self.path_results}exposure_view/adj_matrices/{agg_period}/",
                    exist_ok=True,
                )
                fct.dump_np_array(
                    arr_binary_adj[period_nb][step],
                    f"{self.path_results}exposure_view/adj_matrices/{agg_period}/arr_binary_adj_on_day_{step}.csv",
                )

    def build_adj_matrices(self):

        # ------------
        # exposure view

        self.get_arr_rev_repo_exp_adj()
        self.get_arr_binary_adj()

    def simulate(self):

        # record and store trajectories & parameters used at step 0
        self.save_param()
        self.fill_step()

        print("simulate the repo market")

        # simulate the network
        for _ in tqdm(range(self.nb_steps - 1)):

            # run one step of the network
            self.Network.step_network()

            # debug check memory
            fct.check_memory()

            # record information
            self.fill_step()

            # expoxt record, dump, and plot
            if self.Network.step % self.dump_period == 0:
                self.fill()
                self.dump()
                gx.plot(self)

        # store the final step (if not already done)
        if self.Network.step % self.dump_period != 0:
            self.fill()
            self.dump()
            gx.plot(self)

    def save_param(self):

        # save parameters to a file
        with open(f"{self.path_results}input_parameters.txt", "w") as f:
            f.write(
                f"nb_banks={self.Network.nb_banks} \n"
                f"alpha_init={self.Network.alpha_init} \n"
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

        # print the parameter to terminal
        with open(f"{self.path_results}input_parameters.txt", "r") as f:
            print(f.read())


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
    heavy_plot,
    notice_period,
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
        LCR_mgt_opt=LCR_mgt_opt,
        min_repo_trans_size=min_repo_trans_size,
        notice_period=notice_period,
    )

    # initialize ClassDynamics
    dynamics = ClassDynamics(
        Network,
        nb_steps=nb_steps,
        path_results=result_location,
        dump_period=dump_period,
        plot_period=plot_period,
        cp_option=cp_option,
        heavy_plot=heavy_plot,
    )

    # simulate
    dynamics.simulate()
