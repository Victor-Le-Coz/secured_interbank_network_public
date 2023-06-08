# imports
import emp_preprocessing as ep
import emp_fake_data as ef
import data_mapping as dm

# parameters
path = "./results/empirical_results/"
plot_period = 1000

nb_lines = int(1e6)  # 1 million lines

# # opt 1: get the mmsr exposure view (prepared by NA)
# df_exposures = ef.get_df_exposures(lines=int(1e5),freq="5h")

# opt 2: get df_mmsr unsecured (used for deposits time series)
df_mmsr_unsecured = ef.get_df_mmsr_unsecured(nb_tran=nb_lines, freq="10min")

# get df_mmsr secured (used for transaction and exposure view)
df_mmsr_secured = ef.get_df_mmsr_secured(
    nb_tran=nb_lines, holidays=dm.holidays
)

# build fake finrep data (used for accounting view)
df_finrep = ef.get_df_finrep()


# get dic reverse repo exp adj history
# # opt 1: from exposure (prepared by NA)
# dic_rev_repo_exp_adj = ep.get_dic_rev_repo_exp_adj_from_exposures(df_exposures=df_exposures,path=path, plot_period=plot_period)

# opt 2: directly from mmsr
df_mmsr_secured_clean = ep.get_df_mmsr_secured_clean(
    df_mmsr_secured, holidays=dm.holidays, path=path, compute_tenor=True
)
dic_rev_repo_exp_adj = ep.get_dic_rev_repo_exp_adj_from_mmsr_secured_clean(
    df_mmsr_secured_clean, path=path, plot_period=plot_period
)

# get aggregated adjency matrices
dic_arr_binary_adj = ep.get_dic_arr_binary_adj(
    dic_rev_repo_exp_adj=dic_rev_repo_exp_adj,
    path=path,
    plot_period=plot_period,
)

# get df_rev_repo_trans
df_rev_repo_trans = ep.get_df_rev_repo_trans(df_mmsr_secured_clean, path=path)

# get dic dashed trajectory
df_finrep_clean = ep.get_df_finrep_clean(df_finrep)
dic_dashed_trajectory = ep.get_dic_dashed_trajectory(
    df_finrep_clean, path=path
)
