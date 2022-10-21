import numpy as np
import pandas as pd
from random import choices


def get_df_mmsr(nb_tran):
    df_mmsr = pd.DataFrame(
        index=pd.period_range(
            start="2020-01-01", freq="1h", periods=nb_tran
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
                np.random.rand(nb_tran) * 250, unit="d"
            )
            + pd.period_range(
                start="2020-01-01", freq="1h", periods=nb_tran
            ).to_timestamp(),
            "first_occurence": choices(
                [True, False],
                k=nb_tran,  # first occurence of the reporting of an evergreen transaction repo
            ),
        },
    )
    return df_mmsr


# Set the path of the maintenance period source file
df_maintenance_periods = pd.read_csv(
    "./support/ECB_maintenance_periods.csv", index_col=0
)
