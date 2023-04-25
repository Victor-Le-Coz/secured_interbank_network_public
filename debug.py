# imports
import emp_preprocessing as ep
import emp_fake_data as ef
import emp_metrics as em
import emp_graphics as eg
import networkx as nx
import graphics as gx
import functions as fct
import numpy as np
import pandas as pd

# parameters
agg_periods = [1, 50, 100, 250]
path_results = "./results/general-testing/"

# load data (faster)
dic_obs_adj_cr, dic_obs_matrix_reverse_repo = (
    ep.get_dic_obs_matrix_reverse_repo(),
    ep.get_dic_obs_adj_tr(),
)

# get aggregated adjency matrices
dic_binary_adjs = em.get_binary_adjs(dic_obs_matrix_reverse_repo, agg_periods)

# get jaccard
df_jaccard = em.get_jaccard(dic_binary_adjs)
eg.plot_jaccard_aggregated(df_jaccard, path_results)

# get density
df_density = em.get_density(dic_binary_adjs)
eg.plot_network_density(df_density, path_results)


# get degree distribution

# for the transactions aggregated
(
    dic_in_degree_distribution,
    df_out_degree_distribution,
) = em.get_degree_distribution(dic_binary_adjs)
eg.plot_step_degree_distribution(
    dic_in_degree_distribution,
    df_out_degree_distribution,
    path_results,
    name="degree_distribution_tr",
)

# for the exposures
dic_binary_adjs = em.get_binary_adjs(dic_obs_adj_cr, [1])
(
    dic_in_degree_distribution,
    df_out_degree_distribution,
) = em.get_degree_distribution(dic_binary_adjs)
eg.plot_step_degree_distribution(
    dic_in_degree_distribution,
    df_out_degree_distribution,
    path_results,
    name="degree_distribution_cr",
)

# for the exposures
dic_binary_adjs = em.get_binary_adjs(dic_obs_adj_cr, [1])
(
    dic_in_degree_distribution,
    df_out_degree_distribution,
) = em.get_degree_distribution(dic_binary_adjs)
eg.plot_step_degree_distribution(
    dic_in_degree_distribution,
    df_out_degree_distribution,
    path_results,
    name="degree_distribution_cr",
)


# for the transactions

# build the dic_binary_adjs
dic_binary_adjs = em.get_binary_adjs(dic_obs_matrix_reverse_repo, agg_periods)

# build nx object
bank_network = nx.from_numpy_matrix(
    dic_binary_adjs[50][200], parallel_edges=False, create_using=nx.DiGraph
)

# run cpnet test
sig_c, sig_x, significant, p_value = fct.cpnet_test(bank_network)

# plot
gx.plot_core_periphery(
    bank_network=bank_network,
    sig_c=sig_c,
    sig_x=sig_x,
    path=path_results,
    step="2020-01-05",
    name_in_title="reverse repos",
)
