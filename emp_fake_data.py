import numpy as np
import pandas as pd
from random import choices

freq = "5h"


def get_df_mmsr(nb_tran):
    df_mmsr = pd.DataFrame(
        index=pd.period_range(
            start="2020-01-01", freq=freq, periods=nb_tran
        ).to_timestamp(),
        data={
            "report_agent_lei": choices(
                ["bank_" + str(i) for i in range(10)], k=nb_tran
            ),
            "cntp_lei": choices(
                ["bank_" + str(i) for i in range(15)]
                + ["fund_" + str(i) for i in range(5)],
                k=nb_tran,
            ),
            "trns_nominal_amt": np.random.rand(nb_tran) * 100,
            "maturity_time_stamp": pd.to_timedelta(
                np.random.rand(nb_tran) * 10, unit="d"
            )
            + pd.period_range(
                start="2020-01-01", freq=freq, periods=nb_tran
            ).to_timestamp(),
            "first_occurence": choices(
                [True, False],
                k=nb_tran,  # first occurence of the reporting of an evergreen transaction repo
            ),
            "trns_type": choices(
                ["BORR", "LEND", "BUYI", "SELL"],
                k=nb_tran,  # first occurence of the reporting of an evergreen transaction repo
            ),
        },
    )
    return df_mmsr


def get_df_exposures(lines):
    df_exposures = pd.DataFrame(
        index=range(lines),
        data={
            "borr_lei": choices(
                ["bank_" + str(i) for i in range(10)], k=lines
            ),
            "lend_lei": choices(
                ["bank_" + str(i) for i in range(10)],
                # + ["fund_" + str(i) for i in range(5)],
                k=lines,
            ),
            "exposure": np.random.rand(lines) * 100,
            "Setdate": pd.period_range(
                start="2020-01-01", freq=freq, periods=lines
            ).to_timestamp(),
        },
    )
    return df_exposures


def get_df_finrep():
    df_finrep = pd.DataFrame(
        data={
            "lei": ["bank_" + str(i) for i in range(50)] * 25,
            "date": sorted(
                list(
                    pd.period_range(
                        start="2020-01-01", freq="1y", periods=25
                    ).to_timestamp()
                )
                * 50
            ),
            "total assets": np.random.rand(50 * 25) * 100,
        },
    )
    return df_finrep


# list of the column in FINREP fake data (to be added)
# balance_sheet_data_clean.dta	total_assets
# balance_sheet_data_clean.dta	own_funds
# balance_sheet_data_clean.dta	own_funds_assets
# balance_sheet_data_clean.dta	cash
# balance_sheet_data_clean.dta	cash_assets
# balance_sheet_data_clean.dta	deposits
# balance_sheet_data_clean.dta	deposits_assets
# balance_sheet_data_clean.dta	loans
# balance_sheet_data_clean.dta	loans_assets
# balance_sheet_data_clean.dta	stock_market_sec_gov
# balance_sheet_data_clean.dta	sec_holdings_assets_m
# balance_sheet_data_clean.dta	stock_nominal_sec_gov
# balance_sheet_data_clean.dta	sec_holdings_assets_n
