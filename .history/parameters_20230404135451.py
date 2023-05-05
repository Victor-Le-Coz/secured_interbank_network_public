agg_periods = [1, 50, 100, 250]

# accounting items
assets = [
    "Cash",
    "Securities Usable",
    "Securities Encumbered",
    "Loans",
    "Reverse Repos",
]

liabilities = [
    "Own Funds",
    "Deposits",
    "Repos",
    "MROs",
]

off_bs_items = [
    "Securities Collateral",
    "Securities Reused",
]

accounting_items = assets + liabilities + off_bs_items

# other items
other_items = ["Initial deposits", "Total assets"]

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
matrices = ["adjency", "trust", "binary_adjency"]

# output metrics
output_single_keys = [
    "Av. in-degree",
    "Collateral reuse",
    "Core-Peri. p_val.",
    "Gini",
    "Repos av. maturity",
    "Repos tot. volume",
    "Repos av. volume",
]

output_mlt_keys = [
    "Jaccard index over ",
    "Network density over ",
]

log_output_single_keys = [
    "Repos av. volume",
]

log_output_mlt_keys = [
    "Network density over ",
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
    "Network density",
    "Jaccard index",
    "Raw jaccard index",
]

network_metrics = [
    "Cash tot. volume",
    "Securities Usable tot. volume",
    "Securities Encumbered tot. volume",
    "Loans tot. volume",
    "Reverse Repos tot. volume",
    "Own Funds tot. volume",
    "Deposits tot. volume",
    "Repos tot. volume",
    "MROs tot. volume",
    "Securities Collateral tot. volume",
    "Securities Reused tot. volume",
    "Av. in-degree",
    "Excess Liquidity",
    "Av. nb. of repo transactions ended",
    "Av. volume of repo transactions ended",
    "Repos av. maturity",
    "Gini",
    "Repos min volume",
    "Repos max volume",
    "Repos av. volume",
    "Assets tot. volume",
    "Collateral reuse",
]


bank_metrics = [
    "Cash",
    "Securities Usable",
    "Securities Encumbered",
    "Loans",
    "Reverse Repos",
    "Own Funds",
    "Deposits",
    "Repos",
    "MROs",
    "Securities Collateral",
    "Securities Reused",
    "Excess Liquidity",
    "Av. in-degree",
    "Av. out-degree",
    "Nb. of repo transactions ended",
    "Av. volume of repo transactions ended",
    "Repos av. maturity",
]