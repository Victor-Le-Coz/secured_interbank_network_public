agg_periods = [1, 50, 100, 250]

# The parameter sets the limit to the float precision when running the algorithm, a value lower than this amount is considered as negligible.
float_limit = 1e-10  # required to model a 1000 bilion euros balance sheet

# accounting items
assets = [
    "cash",
    "securities usable",
    "securities encumbered",
    "loans",
    "reverse repo exposures",
]

liabilities = [
    "own funds",
    "deposits",
    "repo exposures",
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
reverse_repos = ["amount", "start_step", "tenor", "status"]

# metrics
metrics = ["degree"]


# output metrics
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

# input parameters
log_input_params = [
    "cash",
    "alpha_pareto",
    "min_repo_size",
    "shocks_vol",
    "n_banks",
]

input_params = [
    "cash",
    "shocks_vol",
    "alpha_pareto",
    "beta",
    "n_banks",
    "min_repo_size",
    "collateral",
]


link_network_metrics = [
    "network density",
    "jaccard index",
    "raw jaccard index",
]


cp_algos = [
    "Rombach",
    "Rossa",
    "LapCore",
    "LapSgnCore",
    "LowRankCore",
    "MINRES",  # do not take weights into acount
    "Surprise",  # do not take weights into acount
    "Lip",  # do not take weights into acount
    "BE",  # do not take weights into acount
]
