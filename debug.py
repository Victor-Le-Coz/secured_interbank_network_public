# imports
import data_preprocessing as dp
import fake_data as fd

# param definition
nb_tran = 100

# load fake data
df_mmsr = fd.get_df_mmsr(nb_tran=100)

dic_obs_adj = dp.build_from_data(df_mmsr=df_mmsr)
