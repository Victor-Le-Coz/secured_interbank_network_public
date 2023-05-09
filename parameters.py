import pandas as pd

# -----------------
# run features

# aggregation periods for the definition of links
agg_periods = [1, 50, 100, 250]

# limit to the float precision
float_limit = 1e-10  # required to model a 1000 bilion euros balance sheet

# ------------------
# variables definition

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

# --------------------
# ploting conventions

small_figsize = (6, 3)  # default
slide_figsize = (12, 6)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots


plt_columns = [
    "item",
    "label",
    "legend",
    "scale",
    "str_format",
    "convertion",
]

# metrics

# accounting view
accounting_metrics = [
    [
        f"{metric}{extension}",
        r"notional (monetary units)",
        f"{metric}{extension}",
        "linear",
        "",
        False,
    ]
    for metric in bank_items
    for extension in [" av. network", " tot. network", ""]
]
collateral_reuse = [
    "collateral reuse",
    r"number of reuse (#)",
    r"number of reuse (#)",
    "linear",
    "",
    False,
]
gini = [
    "gini",
    r"gini (%)",
    r"gini (%)",
    "linear",
    "",
    "%",
]


# transaction view
repo_transactions_maturity_av_network = [
    [
        f"repo transactions maturity{extension}",
        r"maturity (days)",
        r"maturity (days)",
        "linear",
        "",
        False,
    ]
    for extension in [" av. network", " av. bank"]
]
repo_transactions_notional_av_network = [
    [
        f"repo transactions notional{extension}",
        r"notional (monetary units)",
        r"notional (monetary units)",
        "linear",
        "",
        False,
    ]
    for extension in [" av. network", " av. bank"]
]
number_repo_transactions_av_network = [
    [
        f"number repo transactions{extension}",
        r"Nb transactions (#)",
        r"Nb transactions (#)",
        "linear",
        "",
        False,
    ]
    for extension in [" av. network", " av. bank"]
]

# exposure view
repo_exposure_stats = [
    [
        f"repo exposures{extension}",
        r"notional (monetary units)",
        f"repo exposures{extension}",
        "linear",
        "",
        False,
    ]
    for extension in [" min network", " max network", " av. network"]
]
jaccard_index = [
    [
        f"jaccard index-{agg_period}",
        r"Jaccard (%)",
        f"{agg_period} day(s)",
        "linear",
        "",
        "%",
    ]
    for agg_period in agg_periods
]
network_density = [
    [
        f"network density-{agg_period}",
        r"density (%)",
        f"{agg_period} day(s)",
        "linear",
        "",
        "%",
    ]
    for agg_period in agg_periods
]
average_degree = [
    [
        f"av. degree-{agg_period}",
        r"degree (#)",
        f"{agg_period} day(s)",
        "linear",
        "",
        False,
    ]
    for agg_period in agg_periods
]

# input parameters
nb_banks = [
    "nb_banks",
    r"$N$ (log scale)",
    r"$N$ (log scale)",
    "log",
    "",
    False,
]
alpha_init = [
    "alpha_init",
    r"$\alpha_0$ (log scale)",
    r"$\alpha_0$ (log scale)",
    "linear",
    "",
    "%",
]
alpha = [
    "alpha",
    r"$\alpha_0$ (log scale)",
    r"$\alpha_0$ (log scale)",
    "log",
    "",
    "%",
]
beta_init = ["beta_init", r"$\beta_0$", r"$\beta_0$", "linear", "", "%"]
beta_reg = ["beta_reg", r"$\beta$", r"$\beta$", "linear", "", "%"]
alpha_pareto = [
    "alpha_pareto",
    r"alpha patero (log scale)",
    r"alpha patero (log scale)",
    "log",
    "",
    "%",
]
shocks_vol = [
    "shocks_vol",
    r"$\sigma$ (log scale)",
    r"$\sigma$ (log scale)",
    "log",
    "",
    "%",
]
min_repo_trans_size = [
    "min_repo_trans_size",
    r"min repo trans. size (log scale)",
    r"min repo trans. size (log scale)",
    "log",
    "",
    False,
]

# get df_plt
df_plt = pd.DataFrame(
    [
        *accounting_metrics,
        collateral_reuse,
        gini,
        *repo_transactions_maturity_av_network,
        *repo_transactions_notional_av_network,
        *number_repo_transactions_av_network,
        *repo_exposure_stats,
        *jaccard_index,
        *network_density,
        *average_degree,
        nb_banks,
        alpha_init,
        alpha,
        beta_init,
        beta_reg,
        alpha_pareto,
        shocks_vol,
        min_repo_trans_size,
    ],
    columns=plt_columns,
)
df_plt.set_index(["item"], inplace=True)

# -------------------
# list of plots

figures_columns = ["file_name", "items", "extension"]


# accounting view
macro_economic_aggregates = [
    "accounting_view/macro_economic_aggregates",
    [
        "loans",
        "central bank funding",
        "total assets",
        "deposits",
        "excess liquidity",
    ],
    " tot. network",
]
collateral_aggregates = [
    "accounting_view/collateral_aggregates",
    ["securities usable", "securities encumbered"]
    + off_bs_items
    + ["repo balance"],
    " tot. network",
]
collateral_reuse = [
    "accounting_view/collateral_reuse",
    ["collateral reuse"],
    "",
]
gini = [
    "accounting_view/gini",
    ["gini"],
    "",
]


# transaction view
repo_transactions_maturity_av_network = [
    "transaction_view/repo_transactions_maturity_av_network",
    ["repo transactions maturity av. network"],
    "",
]
repo_transactions_notional_av_network = [
    "transaction_view/repo_transactions_notional_av_network",
    ["repo transactions notional av. network"],
    "",
]
number_repo_transactions_av_network = [
    "transaction_view/number_repo_transactions_av_network",
    ["number repo transactions av. network"],
    "",
]


# exposure view
repo_exposure_stats = [
    "exposure_view/repo_exposure_stats",
    [
        "repo exposures min network",
        "repo exposures max network",
        "repo exposures av. network",
    ],
    "",
]
jaccard_index = [
    "exposure_view/jaccard_index",
    [f"jaccard index-{agg_period}" for agg_period in agg_periods],
    "",
]
network_density = [
    "exposure_view/network_density",
    [f"network density-{agg_period}" for agg_period in agg_periods],
    "",
]
average_degree = [
    "exposure_view/average_degree",
    [f"av. degree-{agg_period}" for agg_period in agg_periods],
    "",
]

# get df_figures
df_figures = pd.DataFrame(
    [
        macro_economic_aggregates,
        collateral_aggregates,
        collateral_reuse,
        gini,
        repo_transactions_maturity_av_network,
        repo_transactions_notional_av_network,
        number_repo_transactions_av_network,
        repo_exposure_stats,
        jaccard_index,
        network_density,
        average_degree,
    ],
    columns=figures_columns,
)
df_figures.set_index(["file_name"], inplace=True)


# -----------------------
# miscellaneous parameters

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
