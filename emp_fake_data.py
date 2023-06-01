import numpy as np
import pandas as pd
from random import choices
import parameters as par

freq = "5h"


def get_df_mmsr_secured(nb_tran):
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
                k=nb_tran,
            ),
        },
    )
    return df_mmsr


def get_df_mmsr_unsecured(nb_tran, freq):
    df_mmsr = pd.DataFrame(
        index=pd.period_range(
            start="2020-01-01", freq=freq, periods=nb_tran
        ).to_timestamp(),
        data={
            "report_agent_lei": choices(
                ["bank_" + str(i) for i in range(5)], k=nb_tran
            ),
            "cntp_lei": choices(
                ["bank_" + str(i) for i in range(5)]
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
            "trns_type": choices(
                ["BORR", "LEND", "BUYI", "SELL"],
                k=nb_tran,
            ),
            "instr_type": ["DPST" for i in range(nb_tran)],
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

    dic_data = {
        "lei": ["bank_" + str(i) for i in range(50)] * 25,
        "date": sorted(
            list(
                pd.period_range(
                    start="2020-01-01", freq="1y", periods=25
                ).to_timestamp()
            )
            * 50
        ),
    }

    for bank_item in par.bank_items:
        dic_data.update({bank_item: np.random.rand(50 * 25) * 100})

    df_finrep = pd.DataFrame(
        data=dic_data,
    )

    return df_finrep


def get_dic_dashed_trajectory(df_finrep):
    dic_dashed_trajectory = {}
    plot_days = pd.to_datetime(
        sorted(list(set(df_finrep["date"].dt.strftime("%Y-%m-%d"))))
    )
    for day in plot_days:
        df_banks = (
            df_finrep[df_finrep["date"] == plot_days[0]]
            .set_index("lei")
            .drop("date", axis=1)
        )
        dic_dashed_trajectory.update({day: df_banks})
    return dic_dashed_trajectory


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
