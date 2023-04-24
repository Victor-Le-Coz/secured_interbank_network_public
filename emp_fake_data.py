import numpy as np
import pandas as pd
from random import choices

freq = "1h"


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
                ["bank_" + str(i) for i in range(15)]
                + ["fund_" + str(i) for i in range(5)],
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
            "total_assets": np.random.rand(50 * 25) * 100,
        },
    )
    return df_finrep


# Set the path of the maintenance period source file
df_maintenance_periods = pd.read_csv(
    "./support/ECB_maintenance_periods.csv", index_col=0
)
