import numpy as np
import numpy.ma as ma
import pandas as pd
import pickle5 as pickle


def build_from_data(mmsr_data, maintenance_periods_list):
    """
    input: mmsr_data: filtered on lend and sell 
    
    """
    # create an Numpy array of the unique LEI of the entities from either report agent or counterparties
    leis = pd.unique(mmsr_data[["cntp_lei", "report_agent_lei"]].values.ravel("K"))

    # initialisation of a dictionary of the observed paths
    for step in sorted(list(set(mmsr_data.index.strftime("%Y-%m-%d")))):
        observed_path = {step: pd.DataFrame(columns=leis, index=leis)}

    # create a new column with a flag "end_maintenance_period" for each date
    mmsr_data = pd.merge(
        mmsr_data, maintenance_periods_list, on="trade_date", how="left"
    )

    # building of the matrices and storage in the dictionary observed_path
    for mmsr_index in mmsr_data.index:
        for date_to_maturity in pd.period_range(
            mmsr_index,
            mmsr_data.loc[mmsr_index, "maturity_date"],
            "1d",  # warning not sure of the name of the marurity date timestamp
        ):
            observed_path.update(
                {
                    date_to_maturity: observed_path[date_to_maturity].loc[
                        mmsr_data.loc[mmsr_index, "report_agent_lei"],
                        mmsr_data.loc[mmsr_index, "cntp_lei"],
                    ]
                    + mmsr_data.trns_nominal_amt[mmsr_index]
                }
            )

        # fill-in the information about the end of maintenance periods
        if mmsr_data.end_maintenance_period[mmsr_index] is True:
            observed_path[t].end_maintenance = True
        else:
            observed_path[t].end_maintenance = False


    pickle.dump(
        observed_path,
        open("./observed_path.pickle", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,)

    return observed_path
