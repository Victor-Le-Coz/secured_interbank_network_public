# imports
import emp_preprocessing as ep
import fake_data as fd
import emp_metrics as em
import emp_graphics as eg

# param definition
nb_tran = 10000

# load fake data
df_mmsr = fd.get_df_mmsr(nb_tran=nb_tran)

# build dic of observed data
dic_obs_adj_cr, dic_obs_adj_tr = dp.build_from_data(df_mmsr=df_mmsr)

print(len(dic_obs_adj_cr.keys()))

# build dic of jaccard index
agg_periods = [1, 50, 100, 250]

dic_jaccard = em.compute_jaccard(dic_obs_adj_tr=dic_obs_adj_tr, agg_periods=agg_periods)

print(dic_jaccard)
