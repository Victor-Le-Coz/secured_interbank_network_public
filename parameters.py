import pandas as pd

agg_periods = [1, 50, 100, 250]

# The parameter sets the limit to the float precision when running the algorithm, a value lower than this amount is considered as negligible.
float_limit = 1e-10  # required to model a 1000 bilion euros balance sheet

# define default figure size (but can always be updated for single charts)
small_figsize = (6, 3)  # default one
slide_figsize = (12, 6)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots notably

# accounting items
assets = [
    "cash",
    "securities usable",
    "securities encumbered",
    "loans",
    "reverse repo balance",
]

liabilities = [
    "own funds",
    "deposits",
    "repo balance",
    "central bank funding",
]

off_bs_items = [
    "securities collateral",
    "securities reused",
]

accounting_items = assets + liabilities + off_bs_items

# other items
other_items = [
    "initial deposits",
    "total assets",
    "excess liquidity",
]

# bank items
bank_items = accounting_items + other_items

# shocks modeling approaches
shocks_methods = [
    "bilateral",
    "multilateral",
    "dirichlet",
    "non-conservative",
]

# initialization approaches
initialization_methods = ["constant", "pareto"]

# matrices
matrices = ["adjency", "trust", "binary_adjency", "non-zero_adjency"]

# reverse repos
transaction_cols = ["amount", "start_step", "tenor", "status"]

# core-periphery algorithms
cp_algos = [
    # "KM_ER", # divide by zero error
    # "KM_config", # divide by zero error
    # "Divisive", # divide by zero error
    "Rombach",
    "Rossa",
    # "LapCore",  # generates bug (scipy.sparse.linalg._eigen.arpack.arpack.ArpackError: ARPACK error 3: No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV.)
    "LapSgnCore",
    # "LowRankCore", # generates bug (scipy.sparse.linalg._eigen.arpack.arpack.ArpackError: ARPACK error 3: No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV.)
    "MINRES",  # do not take weights into acount
    # "Surprise",  # do not take weights into acount & too slow
    "Lip",  # do not take weights into acount
    "BE",  # do not take weights into acount
]


# input params
input_params = [
    "cash",
    "shocks_vol",
    "alpha_pareto",
    "beta",
    "n_banks",
    "min_repo_size",
    "collateral",
]

# ploting conventions
plt_columns = [
    "item",
    "label",
    "scale",
    "str_format",
    "convertion",
]
nb_banks = ["nb_banks", r"$N$ (log scale)", "log", "", False]
alpha_init = [
    "alpha_init",
    r"$\alpha_0$ (log scale)",
    "linear",
    "",
    "%",
]
alpha = ["alpha", r"$\alpha_0$ (log scale)", "log", "", "%"]
beta_init = ["beta_init", r"$\beta_0$", "linear", "", "%"]
beta_reg = ["beta_reg", r"$\beta$", "linear", "", "%"]
beta_star = ["beta_reg", r"$\beta*$", "linear", "", "%"]
gamma = ["gamma", r"$\gamma$", "linear", "", "%"]
alpha_pareto = ["alpha_pareto", r"alpha patero (log scale)", "log", "", "%"]
shocks_vol = ["shocks_vol", r"$\sigma$ (log scale)", "log", "", "%"]
min_repo_trans_size = [
    "min_repo_trans_size",
    r"min repo trans. size (log scale)",
    "log",
    "",
    False,
]
# add here any other metric or item for which a ploting convention must be defined


df_plt = pd.DataFrame(
    [
        alpha,
        beta_init,
        beta_reg,
        beta_star,
        gamma,
        alpha_pareto,
        shocks_vol,
        min_repo_trans_size,
    ],
    columns=plt_columns,
)
df_plt.set_index(["item"], inplace=True)


# ---------------------------------
# to be reviewed:
log_input_params = [
    "cash",
    "alpha_pareto",
    "min_repo_size",
    "shocks_vol",
    "n_banks",
]
output_single_keys = [
    "av. in-degree",
    "collateral reuse",
    "core-peri. p-val.",
    "gini",
    "repos av. maturity",
    "repos tot. network",
    "repos av. volume",
]
output_mlt_keys = [
    "jaccard index over ",
    "network density over ",
]

log_output_single_keys = [
    "repos av. volume",
]

log_output_mlt_keys = [
    "network density over ",
]

link_network_metrics = [
    "network density",
    "jaccard index",
    "raw jaccard index",
]
