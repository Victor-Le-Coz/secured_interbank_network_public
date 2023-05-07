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
        fct.init_results_path(self.path_results)

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
        self.arr_reverse_repo_adj = np.zeros(
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

    def step_record_trajectories(self):
        self.fill_df_network_trajectory()
        self.fill_df_bank_trajectory()
        self.fill_arr_total_assets()

    def fill_df_network_trajectory(self):
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

        # repo exposures
        if len(self.Network.dic_matrices["non-zero_adjency"]) == 0:
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures min network"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures max network"
            ] = 0
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures av. network"
            ] = 0
        else:
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures min network"
            ] = np.min(self.Network.dic_matrices["non-zero_adjency"])
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures max network"
            ] = np.max(self.Network.dic_matrices["non-zero_adjency"])
            self.df_network_trajectory.loc[
                self.Network.step, "repo exposures av. network"
            ] = np.mean(self.Network.dic_matrices["non-zero_adjency"])

    def fill_df_bank_trajectory(self):

        # Build the time series of the accounting items
        for item in par.bank_items:
            self.df_bank_trajectory.loc[
                self.Network.step, item
            ] = self.Network.df_banks.loc[self.single_bank_id, item]

    def fill_arr_total_assets(self):
        self.arr_total_assets[self.Network.step] = np.array(
            self.Network.df_banks["total assets"]
        )

    def expost_record_trajectories(self):

        # build the df_reverse_repos history from all banks data
        self.Network.build_df_reverse_repos()

        # build adj matrices from df_reverse repos
        self.build_adj_matrices()

        # fill step df_network_trajectory & df_bank_trajectory from the df_reverse_repos data
        print("expost fill step df network trajectory")
        for step in tqdm(
            range(self.Network.step + 1)
        ):  # +1 to cover all steps up to now
            self.expost_fill_step_df_network_trajectory(step)
            self.expost_fill_step_df_bank_trajectory(step)

        # fill df_network_trajectory & df_bank_trajectory from the adj matrices
        self.expost_fill_dic_degree()
        self.expost_fill_df_network_trajectory()
        self.expost_fill_df_bank_trajectory()

        # save the data frame results
        self.df_bank_trajectory.to_csv(
            f"{self.path_results}df_bank_trajectory.csv"
        )
        self.df_network_trajectory.to_csv(
            f"{self.path_results}df_network_trajectory.csv"
        )
        self.Network.store_network(self.path_results)

    def expost_fill_step_df_network_trajectory(self, step):

        # very slow, parcours de df

        df = self.Network.df_reverse_repos

        # repos maturity av. network
        df_ending = df[df["tenor"] + df["start_step"] == step - 1]
        if df_ending["amount"].sum() > 0:
            self.df_network_trajectory.loc[
                step, "repos maturity av. network"
            ] = (df_ending["amount"] @ df_ending["tenor"]) / df_ending[
                "amount"
            ].sum()
        else:
            self.df_network_trajectory.loc[
                step, "repos maturity av. network"
            ] = 0

        # amount_ending_starting av. network
        df_starting = df[df["start_step"] == step - 1]
        nb_trans = len(df_ending) + len(df_starting)
        if nb_trans > 0:
            self.df_network_trajectory.loc[
                step,
                "amount_ending_starting av. network",
            ] = (
                df_ending["amount"].sum() + df_starting["amount"].sum()
            ) / nb_trans
        else:
            self.df_network_trajectory.loc[
                step,
                "amount_ending_starting av. network",
            ] = 0

    def expost_fill_df_network_trajectory(self):

        # expost jaccard
        df_jaccard = em.get_jaccard(
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

        # expost

    def expost_fill_df_bank_trajectory(self):

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

    def expost_fill_dic_degree(self):

        # degree distribution
        (
            self.dic_in_degree,
            self.dic_out_degree,
            self.dic_degree,
        ) = em.get_degree_distribution(
            self.dic_arr_binary_adj,
            path=f"{self.path_results}degree_distribution/",
        )

    def expost_fill_step_df_bank_trajectory(self, step):

        df = self.Network.df_reverse_repos

        # check of the single bank entered into any reverse repos
        if self.single_bank_id in df.index.get_level_values(0):
            df = df.loc[self.single_bank_id]
        else:
            self.df_bank_trajectory.loc[step, "repos maturity av. bank"] = 0
            self.df_bank_trajectory.loc[
                step,
                "amount_ending_starting av. bank",
            ] = 0

        # repos maturity av. network
        df_ending = df[df["tenor"] + df["start_step"] == step - 1]
        if df_ending["amount"].sum() > 0:
            self.df_bank_trajectory.loc[step, "repos maturity av. bank"] = (
                df_ending["amount"] @ df_ending["tenor"]
            ) / df_ending["amount"].sum()
        else:
            self.df_bank_trajectory.loc[step, "repos maturity av. bank"] = 0

        # amount_ending_starting av. network
        df_starting = df[df["start_step"] == step - 1]
        nb_trans = len(df_ending) + len(df_starting)
        if nb_trans > 0:
            self.df_bank_trajectory.loc[
                step,
                "amount_ending_starting av. bank",
            ] = (
                df_ending["amount"].sum() + df_starting["amount"].sum()
            ) / nb_trans
        else:
            self.df_bank_trajectory.loc[
                step,
                "amount_ending_starting av. bank",
            ] = 0

        # nb_ending starting per signle bank
        self.df_bank_trajectory.loc[
            step,
            "nb_ending_starting",
        ] = nb_trans

    def build_arr_reverse_repo_adj_from_df_reverse_repos(self):

        # print
        print("build arr_reverse_repo_adj from df_reverse_repos")

        # loop over the rows of df_reverse_repos transactions
        for index, row in tqdm(self.Network.df_reverse_repos.iterrows()):

            # get the tenor (fill by current step if tenor is empty)
            if np.isnan(row["tenor"]):
                tenor = self.Network.step - row["start_step"]
            else:
                tenor = row["tenor"]

            # add the amont of the transaction over its tenor
            for step in range(
                row["start_step"], row["start_step"] + tenor + 1
            ):
                self.arr_reverse_repo_adj[step, index[0], index[1]] += row[
                    "amount"
                ]

        # loop over the steps to clean values close to zero
        for step in range(self.Network.step + 1):
            self.arr_reverse_repo_adj[step][
                self.arr_reverse_repo_adj[step] < self.Network.min_repo_size
            ] = 0

        # save last step to csv
        fct.init_path(f"{self.path_results}matrices/")
        fct.dump_np_array(
            self.arr_reverse_repo_adj[self.Network.step],
            f"{self.path_results}matrices/arr_reverse_repo_adj_{self.Network.step}.csv",
        )

    def build_arr_binary_adj(self):

        # print
        print("build arr_binary_adj (compiled)")

        # convert list to array
        arr_agg_period = np.array(par.agg_periods)

        # build arr of results with numba
        arr_binary_adj = ep.fast_build_arr_binary_adj(
            self.arr_reverse_repo_adj, arr_agg_period, self.Network.step
        )

        # loop over agg periods
        for period_nb, agg_period in enumerate(par.agg_periods):

            # convert array results to dictionaries
            self.dic_arr_binary_adj[agg_period] = arr_binary_adj[period_nb]

            # save last step to csv
            fct.init_path(f"{self.path_results}matrices/{agg_period}/")
            fct.dump_np_array(
                arr_binary_adj[period_nb][self.Network.step],
                f"{self.path_results}matrices/{agg_period}/arr_binary_adj_on_day_{self.Network.step}.csv",
            )

    def build_adj_matrices(self):
        self.build_arr_reverse_repo_adj_from_df_reverse_repos()
        self.build_arr_binary_adj()

    def simulate(self, output_keys=False):

        # record and store trajectories & parameters used at step 0
        self.save_param()
        self.step_record_trajectories()

        print("simulate the repo market")

        # simulate the network
        for _ in tqdm(range(self.nb_steps - 1)):

            # run one step of the network
            self.Network.step_network()

            # record trajectories
            self.step_record_trajectories()

            # compute and dump expost trajectories every dump period
            if self.Network.step % self.dump_period == 0:
                self.expost_record_trajectories()
                self.Graphics.plot_all_trajectories()

        # store the final step (if not already done)
        if self.Network.step % self.dump_period != 0:
            self.expost_record_trajectories()
            self.Graphics.plot_all_trajectories()

        # final print
        self.Graphics.plot_final_step()
        self.print_summary()

        if output_keys:
            output = self.build_output(output_keys)
            return output

    def print_summary(self):
        for metric in [
            "repos maturity av. network",
            "repo exposures av. network",
        ]:
            print(
                f"{metric}:{self.df_network_trajectory.loc[self.Network.step, metric]}"
            )

    def save_param(self):
        with open(self.path_results + "param.txt", "w") as f:
            f.write(
                (
                    "nb_banks={} \n"
                    "alpha={} \n"
                    "beta_init={} \n"
                    "beta_reg={} \n"
                    "beta_star={} \n"
                    "gamma={} \n"
                    "collateral_value={} \n"
                    "initialization_method={} \n"
                    "alpha_pareto={} \n"
                    "shock_method={} \n"
                    "shocks_law={} \n"
                    "shocks_vol={} \n"
                    "result_location={} \n"
                    "min_repo_size={} \n"
                    "nb_steps={} \n"
                    "save_every={} \n"
                    "jaccard_periods={} \n"
                    "LCR_mgt_opt={} \n"
                ).format(
                    self.Network.nb_banks,
                    self.Network.alpha,
                    self.Network.beta_init,
                    self.Network.beta_reg,
                    self.Network.beta_star,
                    self.Network.gamma,
                    self.Network.collateral_value,
                    self.Network.initialization_method,
                    self.Network.alpha_pareto,
                    self.Network.shocks_method,
                    self.Network.shocks_law,
                    self.Network.shocks_vol,
                    self.path_results,
                    self.Network.min_repo_size,
                    self.nb_steps,
                    self.dump_period,
                    par.agg_periods,
                    self.Network.LCR_mgt_opt,
                )
            )

    def build_output(self):
        output = {}
        stat_len_step = 250

        # build the time series metrics outputs
        for metric in self.df_network_trajectory.columns:
            output.update(
                {
                    metric: self.df_network_trajectory.loc[
                        -stat_len_step:, metric
                    ].mean()
                }
            )

        output.update({"core-peri. p-val.": self.p_value})

        return output


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
    min_repo_size,
    nb_steps,
    dump_period,
    plot_period,
    cp_option,
    LCR_mgt_opt,
    output_keys,
):

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
        min_repo_size=min_repo_size,
        LCR_mgt_opt=LCR_mgt_opt,
    )

    dynamics = ClassDynamics(
        Network,
        nb_steps=nb_steps,
        path_results=result_location,
        dump_period=dump_period,
        plot_period=plot_period,
        cp_option=cp_option,
    )

    output = dynamics.simulate(
        output_keys=output_keys,
    )

    return output
