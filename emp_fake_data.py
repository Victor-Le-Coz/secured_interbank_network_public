import numpy as np
import pandas as pd
from random import choices
import parameters as par


def get_df_mmsr_secured(nb_tran, freq="5h"):
    df_mmsr = pd.DataFrame(
        index=range(nb_tran),
        data={
            "trade_date": pd.period_range(
                start="2020-01-01", freq=freq, periods=nb_tran
            ).to_timestamp(),
            "unique_trns_id": range(nb_tran),
            "proprietary_trns_id": choices(
                ["bank_" + str(i) for i in range(10)], k=nb_tran
            ),
            "cntp_proprietary_trns_id": choices(
                ["bank_" + str(i) for i in range(15)]
                + ["fund_" + str(i) for i in range(5)],
                k=nb_tran,
            ),
            "trns_nominal_amt": np.random.rand(nb_tran) * 100,
            "maturity_date": pd.to_timedelta(
                np.random.rand(nb_tran) * 10, unit="d"
            )
            + pd.period_range(
                start="2020-01-01", freq=freq, periods=nb_tran
            ).to_timestamp(),
            "maturity_bucket": choices(
                [
                    "NA",
                    "O/N",
                    "T/N",
                    "S/N",
                    "1W",
                    "2W",
                    "1M",
                    "2M",
                    "3M",
                    "6M",
                    "9M",
                    "12M",
                ],
                k=nb_tran,
            ),
            "trns_type": choices(
                ["BORR", "LEND", "BUYI", "SELL"],
                k=nb_tran,
            ),
        },
    )
    return df_mmsr


def get_df_mmsr_unsecured(nb_tran, freq="5h"):
    df_mmsr = pd.DataFrame(
        index=range(nb_tran),
        data={
            "trade_date": pd.period_range(
                start="2020-01-01", freq=freq, periods=nb_tran
            ).to_timestamp(),
            "unique_trns_id": range(nb_tran),
            "proprietary_trns_id": choices(
                ["bank_" + str(i) for i in range(10)], k=nb_tran
            ),
            "cntp_proprietary_trns_id": choices(
                ["bank_" + str(i) for i in range(15)]
                + ["fund_" + str(i) for i in range(5)],
                k=nb_tran,
            ),
            "trns_nominal_amt": np.random.rand(nb_tran) * 100,
            "maturity_date": pd.to_timedelta(
                np.random.rand(nb_tran) * 10, unit="d"
            )
            + pd.period_range(
                start="2020-01-01", freq=freq, periods=nb_tran
            ).to_timestamp(),
            "MATURITY_BUCKET_CD": choices(
                [
                    "NA",
                    "O/N",
                    "T/N",
                    "S/N",
                    "1W",
                    "2W",
                    "1M",
                    "2M",
                    "3M",
                    "6M",
                    "9M",
                    "12M",
                ],
                k=nb_tran,
            ),
            "trns_type": choices(
                ["BORR", "LEND", "BUYI", "SELL"],
                k=nb_tran,
            ),
            "instr_type": ["DPST" for i in range(nb_tran)],
        },
    )
    return df_mmsr


def get_df_exposures(lines, freq="5h"):
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
