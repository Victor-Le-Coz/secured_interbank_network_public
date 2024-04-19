import numpy as np
import pandas as pd
from random import choices
import parameters as par
import data_mapping as dm


def get_df_mmsr_secured(nb_tran, holidays):

    # define the european mmsr calendar
    bbday = pd.offsets.CustomBusinessDay(holidays=holidays)

    # define the days on which mmsr trasaction can occure
    days = pd.bdate_range(
        "2016-01-08", "2023-01-01", freq="C", holidays=dm.holidays
    )

    # define the set of existing collateral isin codes
    isins = [f"isin_{i}" for i in range(50)]

    # build the mmsr data frame
    df_mmsr = pd.DataFrame(
        data={
            "trade_date": choices(days, k=nb_tran),
            "unique_trns_id": range(nb_tran),
            "report_agent_lei": choices(
                ["bank_" + str(i) for i in range(10)], k=nb_tran
            ),
            "cntp_lei": choices(
                ["bank_" + str(i) for i in range(15)],
                # + ["fund_" + str(i) for i in range(5)],
                k=nb_tran,
            ),
            "trns_nominal_amt": np.random.rand(nb_tran) * 100,
            "maturity_band": choices(
                dm.maturity_band, k=nb_tran
            ),  # WARNING inconstent with the maturity date
            "trns_type": choices(
                ["BORR", "LEND", "BUYI", "SELL"],
                k=nb_tran,
            ),
            "coll_isin": choices(isins, k=nb_tran),
        },
    )

    # convert to datetime (stange error when building dates from choices())
    df_mmsr["trade_date"] = pd.to_datetime(df_mmsr["trade_date"])

    # build the maturity date # WARNING inconstent with the maturity band
    apply_func = (
        lambda row: row["trade_date"] + int(np.random.rand(1) * 50) * bbday
    )
    df_mmsr["maturity_date"] = df_mmsr.apply(apply_func, axis=1)

    # create some evergreens repeating up to x days
    X = 50
    for nb_repetting_days in range(2, X):

        # define the start and end of the selection of lines in df MMSR
        block_size = int(nb_tran / (X * 2))
        start = (nb_repetting_days - 1) * block_size
        end = nb_repetting_days * block_size  # 50% of evergreen

        # choose where to cut the evergreen into 2 segments
        cut = int(nb_repetting_days / 2)

        for row in range(1, nb_repetting_days):

            if not (
                row in [cut, cut + 1, cut + 2, cut + 3, cut + 5]
            ):  # allow up to 5 days without reporting

                # take the lines from df_mmsr
                df_evergreen = df_mmsr.iloc[start:end]

                # move up the maturity and trade date
                if holidays:
                    df_evergreen["trade_date"] = (
                        df_evergreen["trade_date"] + row * bbday
                    )
                    df_evergreen["maturity_date"] = (
                        df_evergreen["maturity_date"] + row * bbday
                    )

                else:
                    df_evergreen["trade_date"] = df_evergreen[
                        "trade_date"
                    ] + pd.Timedelta(days=row)
                    df_evergreen["maturity_date"] = df_evergreen[
                        "maturity_date"
                    ] + pd.Timedelta(days=row)

                # add the lines to df_mmsr
                df_mmsr = pd.concat([df_mmsr, df_evergreen], axis=0)

    df_mmsr["settlement_date"] = df_mmsr["trade_date"]

    df_mmsr.reset_index(inplace=True, drop=True)

    return df_mmsr


def get_df_mmsr_unsecured(nb_tran, holidays):

    # define the european mmsr calendar
    bbday = pd.offsets.CustomBusinessDay(holidays=holidays)

    # define the days on which mmsr trasaction can occure
    days = pd.bdate_range(
        "2016-01-08", "2023-01-01", freq="C", holidays=dm.holidays
    )

    df_mmsr = pd.DataFrame(
        index=range(nb_tran),
        data={
            "trade_date": choices(days, k=nb_tran),
            "unique_trns_id": range(nb_tran),
            "report_agent_lei": choices(
                ["bank_" + str(i) for i in range(10)], k=nb_tran
            ),
            "cntp_lei": choices(
                ["bank_" + str(i) for i in range(15)]
                + ["fund_" + str(i) for i in range(5)],
                k=nb_tran,
            ),
            "trns_nominal_amt": np.random.rand(nb_tran) * 100,
            "trns_type": choices(
                ["BORR", "LEND", "BUYI", "SELL"],
                k=nb_tran,
            ),
            "instr_type": ["DPST" for i in range(nb_tran)],
        },
    )

    # convert to datetime (stange error when building dates from choices())
    df_mmsr["trade_date"] = pd.to_datetime(df_mmsr["trade_date"])

    # build the maturity date # WARNING inconstent with the maturity band
    apply_func = (
        lambda row: row["trade_date"] + int(np.random.rand(1) * 50) * bbday
    )
    df_mmsr["maturity_date"] = df_mmsr.apply(apply_func, axis=1)

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
                start="2000-01-03", freq=freq, periods=lines
            ).to_timestamp(),
        },
    )
    return df_exposures


def get_df_finrep():
    dic_data = {
        "report_agent_lei": ["bank_" + str(i) for i in range(50)]
        * 40,  # comes from a maping of FINREP lei to MMSR lei
        "qdate": sorted(
            list(
                pd.period_range(
                    start="2000-01-03", freq="1y", periods=40
                ).to_timestamp()
            )
            * 50
        ),
    }

    for bank_item in par.bank_items:
        dic_data.update({bank_item: np.random.rand(50 * 40) * 100})

    df_finrep = pd.DataFrame(
        data=dic_data,
    )

    return df_finrep
