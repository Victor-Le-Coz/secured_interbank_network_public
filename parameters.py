import pandas as pd
import data_mapping as dm

def list_exclusion(list1, list2):
    return [x for x in list1 if x not in set(list2)]

# -----------------
# run features

# aggregation periods for the definition of links
agg_periods = [1, 50, 100, 250]

# limit to the float precision
float_limit = 1e-9  # required to model a 1000 billion euros balance sheet

# nb of days on which stationary average is computed
len_statio = 200

# print each metric run name
detailed_prints = False

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
    "total liabilities",
    "borrowings"
]
finrep_items = list_exclusion(
    list(dm.dic_finrep_columns.values()), accounting_items + other_items
)

# bank items
# bank_items = accounting_items + other_items + finrep_items # for empirical data
bank_items = accounting_items + other_items # for empirical data

regulatory_ratios = ["reserve ratio", "liquidity ratio", "leverage ratio"]


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
transaction_cols = ["amount", "start_step", "end_step", "status"]

# -----------------------
# miscellaneous parameters

# core-periphery algorithms: https://github.com/skojaku/core-periphery-detection
cp_algos = [
    # "KM_ER",  # divide by zero error
    # "KM_config",  # divide by zero error
    # "Divisive",  # divide by zero error - generates a strange bug, kills the whole computation even with the try except option
    # "Rombach",
    # "Rossa",
    # "LapCore",  # generates bug (scipy.sparse.linalg._eigen.arpack.arpack.ArpackError: ARPACK error 3: No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV.)
    # "LapSgnCore",
    # "LowRankCore",  # generates bug (scipy.sparse.linalg._eigen.arpack.arpack.ArpackError: ARPACK error 3: No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV.)
    # "MINRES",  # do not take weights into acount
    # "Surprise",  # do not take weights into acount & too slow
    "Lip",  # do not take weights into acount - The one chosen by LUX !
    # "BE",  # do not take weights into acount
]

# power law analysis
powerlaw_bank_items = ["total assets"]

# for plots per bank using empirical data
bank_ids = [f"bank_{i}" for i in range(150)]

# --------------------
# ploting conventions

small_figsize = (3, 3)  # default
large_figsize = (6, 3)  # default
slide_figsize = (12, 15)  # for the single trajectories
halfslide_figsize = (6, 6)  # for the network plots


plt_columns = [
    "item",
    "label",
    "legend",
    "scale",
    "style",
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
    for extension in [
        " av. network",
        " tot. network",
        "",
    ]
]

regulatory_ratios_metrics = [
    [
        f"{metric}{extension}",
        r"ratio (%)",
        f"{metric}{extension}",
        "log",
        "",
        False,
    ]
    for metric in regulatory_ratios
    for extension in [
        " av. network",
        "",
    ]
]
collateral_reuse = [
    "collateral reuse",
    r"number of reuse (#)",
    r"number of reuse (#)",
    "linear",
    "",
    False,
]

borrowings_ov_deposits = [
    "borrowings ov. deposits tot. network",
    r"ratio (%)",
    r"borrowings ov. deposits tot. network (%)",
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
        f"repo transactions maturity{extension}",
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
        f"repo transactions notional{extension}",
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
        f"number repo transactions{extension}",
        "linear",
        "",
        False,
    ]
    for extension in [" av. network", " av. bank"]
]

# exposure view
stat_extensions = [" min network", " max network", " av. network"]
repo_exposure_stats = [
    [
        f"repo exposures{extension}",
        r"notional (monetary units)",
        f"repo exposures{extension}",
        "linear",
        "",
        False,
    ]
    for extension in stat_extensions
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
degree_stats = [
    [
        f"degree{extension}-{agg_period}",
        r"degree (#)",
        f"{extension}-{agg_period} day(s)",
        "linear",
        "",
        False,
    ]
    for agg_period in agg_periods
    for extension in stat_extensions
]
cpnet_pvalue = [
    [
        f"cpnet p-value {algo}-{agg_period}",
        r"core-periphery p-value",
        f"{algo}-{agg_period} day(s)",
        "symlog",
        ".-",
        "",
    ]
    for algo in cp_algos
    for agg_period in agg_periods + ["weighted"]
]
powerlaw_alpha = [
    [
        f"powerlaw alpha {bank_item}{extension}",
        r"power law alpha",
        f"{bank_item}{extension}",
        "linear",
        ".-",
        "",
    ]
    for bank_item in powerlaw_bank_items
    for extension in ["", " over total assets"]
]
benchmark_laws = ["exponential", "lognormal", "power_law"]
powerlaw_pvalue = [
    [
        f"powerlaw p-value {benchmark_law} {bank_item}{extension}",
        r"power law p-value " + benchmark_law,
        f"{bank_item}{extension}",
        "log",
        ".-",
        "",
    ]
    for benchmark_law in benchmark_laws
    for bank_item in powerlaw_bank_items
    for extension in ["", " over total assets"]
]

powerlaw_direction = [
    [
        f"powerlaw direction {benchmark_law} {bank_item}{extension}",
        r"power law direction " + benchmark_law,
        f"{bank_item}{extension}",
        "linear",
        ".-",
        "",
    ]
    for benchmark_law in benchmark_laws
    for bank_item in powerlaw_bank_items
    for extension in ["", " over total assets"]
]

# input parameters
nb_banks = [
    "nb_banks",
    r"$N$ (#, log scale)",
    r"$N$ (#, log scale)",
    "log",
    "",
    False,
]
initial_deposits_size = [
    "initial_deposits_size",
    r"av. initial deposits per bank (monetary units, log scale)",
    r"av. initial deposits per bank (monetary units, log scale)",
    "log",
    "",
    False,
]
alpha_init = [
    "alpha_init",
    r"$\alpha_0$ (%)",
    r"$\alpha_0$ (%)",
    "linear",
    "",
    "%",
]
alpha = [
    "alpha",
    r"$\alpha$ (%)",
    r"$\alpha$ (%)",
    "linear",
    "",
    "%",
]
beta_init = [
    "beta_init",
    r"$\beta_0$ (%)",
    r"$\beta_0$ (%)",
    "linear",
    "",
    "%",
]
beta_reg = [
    "beta_reg",
    r"$\beta$ (%)",
    r"$\beta$ (%)",
    "linear",
    "",
    "%"
]
gamma_init = [
    "gamma_init",
    r"$\gamma_0$ (%)",
    r"$\gamma_0$ (%)",
    "linear",
    "",
    "%",
]
gamma = [
    "gamma",
    r"$\gamma$ (%)",
    r"$\gamma$ (%)",
    "linear",
    "",
    "%",
]
gamma_star = [
    "gamma_star",
    r"$\gamma^*$ (%)",
    r"$\gamma^*$ (%)",
    "linear",
    "",
    "%",
]
gamma_new = [
    "gamma_new",
    r"$\gamma_{\mathrm{new}}$ (%)",
    r"$\gamma_{\mathrm{new}}$ (%)",
    "linear",
    "",
    "%",
]

alpha_pareto = [
    "alpha_pareto",
    r"$\nu$ (log scale)",
    r"$\nu$  (log scale)",
    "log",
    "",
    False,
]
shocks_vol = [
    "shocks_vol",
    r"$\sigma$ (%)",
    r"$\sigma$ (%)",
    "linear",
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
learning_speed = [
    "learning_speed",
    r"$\lambda$",
    r"$\lambda$",
    "log",
    "",
    False,
]

test_formating = ["dingo 1", " dingo ", " dingo", " dingo ", " dingo ", " dingo ", " dingo ", " dingo ", " dingo ", " dingo ", " dingo ", " dingo ", " dingo ", " dingo "]

# get df_plt
df_plt = pd.DataFrame(
    [
        *accounting_metrics,
        *regulatory_ratios_metrics,
        collateral_reuse,
        borrowings_ov_deposits,
        gini,
        *repo_transactions_maturity_av_network,
        *repo_transactions_notional_av_network,
        *number_repo_transactions_av_network,
        *repo_exposure_stats,
        *jaccard_index,
        *network_density,
        *degree_stats,
        *cpnet_pvalue,
        *powerlaw_alpha,
        *powerlaw_pvalue,
        *powerlaw_direction,
        nb_banks,
        initial_deposits_size,
        alpha_init,
        alpha,
        beta_init,
        beta_reg,
        gamma_init,
        gamma,
        gamma_new,
        alpha_pareto,
        shocks_vol,
        min_repo_trans_size,
        learning_speed,
    ],
    columns=plt_columns,
)
df_plt.set_index(["item"], inplace=True)

# -------------------
# list of plots

figures_columns = ["file_name", "items", "extension"]


# accounting view
fig_macro_economic_aggregates = [
    "accounting_view/macro_economic_aggregates",
    [
        "loans",
        "central bank funding",
        "total assets",
        "deposits",
        "excess liquidity",
        "borrowings",
    ],
    " tot. network",
]
fig_collateral_aggregates = [
    "accounting_view/collateral_aggregates",
    ["securities usable", "securities encumbered"]
    + off_bs_items
    + ["repo balance"],
    " tot. network",
]
fig_collateral_reuse = [
    "accounting_view/collateral_reuse",
    ["collateral reuse"],
    "",
]
fig_borrowings_ov_deposits = [
    "accounting_view/borrowings_ov_deposits",
    ["borrowings ov. deposits tot. network"],
    "",
]
fig_gini = [
    "accounting_view/gini",
    ["gini"],
    "",
]
fig_regulatory_ratios = [
    "accounting_view/regulatory_ratios",
    regulatory_ratios,
    " av. network",
]


# transaction view
fig_repo_transactions_maturity_av_network = [
    "transaction_view/repo_transactions_maturity_av_network",
    ["repo transactions maturity av. network"],
    "",
]
fig_repo_transactions_notional_av_network = [
    "transaction_view/repo_transactions_notional_av_network",
    ["repo transactions notional av. network"],
    "",
]
fig_number_repo_transactions_av_network = [
    "transaction_view/number_repo_transactions_av_network",
    ["number repo transactions av. network"],
    "",
]


# exposure view
fig_repo_exposure_stats = [
    "exposure_view/repo_exposure_stats",
    [
        "repo exposures min network",
        "repo exposures max network",
        "repo exposures av. network",
    ],
    "",
]
fig_jaccard_index = [
    "exposure_view/jaccard_index",
    [f"jaccard index-{agg_period}" for agg_period in agg_periods],
    "",
]
fig_network_density = [
    "exposure_view/network_density",
    [f"network density-{agg_period}" for agg_period in agg_periods],
    "",
]
fig_degree_stats = [
    "exposure_view/degree_stats",
    [
        f"degree{extension}-{agg_period}"
        for extension in stat_extensions
        for agg_period in agg_periods
    ],
    "",
]
figs_cpnet_pvalues = [
    [
        f"exposure_view/core-periphery/cpnet_pvalue-{agg_period}",
        [
            f"cpnet p-value {algo}-{agg_period}"  # extensions have normaly a space, here we add it in front of algo (which have no space)
            for algo in cp_algos
        ],
        "",
    ]
    for agg_period in agg_periods + ["weighted"]
]  # create one figure per agg period

figs_cpnet_pvalues_opt = [
    [
        f"exposure_view/core-periphery/cpnet_pvalue-{algo}",
        [
            f"cpnet p-value {algo}-{agg_period}"  # extensions have normaly a space, here we add it in front of algo (which have no space)
            for agg_period in agg_periods + ["weighted"]
        ],
        "",
    ]
    for algo in cp_algos
]  # create one figure per algo

fig_powerlaw_alpha = [
    f"accounting_view/power_law/powerlaw_alpha",
    [
        f"powerlaw alpha {bank_item}"  # extensions have normaly a space, here we add it in front of algo (which have no space)
        for bank_item in bank_items
    ],
    "",
]
fig_powerlaw_pvalue = [
    f"accounting_view/power_law/powerlaw_pvalue",
    [
            f"{metric} {bank_item}"
            for metric in [
                f"{ind} {benchmark_law}"
                for ind in ["powerlaw direction", "powerlaw p-value"]
                for benchmark_law in benchmark_laws
            ]
            for bank_item in bank_items
        ],
    "",
]


# get df_figures
df_figures = pd.DataFrame(
    [
        fig_macro_economic_aggregates,
        fig_collateral_aggregates,
        fig_regulatory_ratios,
        fig_collateral_reuse,
        fig_borrowings_ov_deposits,
        fig_gini,
        fig_repo_transactions_maturity_av_network,
        fig_repo_transactions_notional_av_network,
        fig_number_repo_transactions_av_network,
        fig_repo_exposure_stats,
        fig_jaccard_index,
        fig_network_density,
        fig_degree_stats,
        # fig_powerlaw_alpha,
        # fig_powerlaw_pvalue,
    ]
    + figs_cpnet_pvalues_opt,
    columns=figures_columns,
)
df_figures.set_index(["file_name"], inplace=True)
