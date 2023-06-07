import datetime
import numpy as np

dic_finrep_columns = {
    "total_assets": "total assets",
    "own_funds": "own funds",
    "cash": "cash",
    "deposits": "deposits",
    "loans": "loans",
    "stock_market_sec_gov": "securities collateral",
}

# list of the column in FINREP fake data (to be added)
# balance_sheet_data_clean.dta	"own_funds_assets" # qu'est ce que c'est ?
# balance_sheet_data_clean.dta	"cash_assets" # qu'est ce que c'est ?
# balance_sheet_data_clean.dta	"deposits_assets" # qu'est ce que c'est ?
# balance_sheet_data_clean.dta	loans_assets # qu'est ce que c'est ?
# balance_sheet_data_clean.dta	"stock_market_sec_gov"# qu'est ce que c'est ?
# balance_sheet_data_clean.dta	"sec_holdings_assets_m"# qu'est ce que c'est ?
# balance_sheet_data_clean.dta	"stock_nominal_sec_gov"# qu'est ce que c'est ?
# balance_sheet_data_clean.dta	"sec_holdings_assets_n" # qu'est ce que c'est ?


holidays = [
    datetime.date(2000, 1, 1),
    datetime.date(2000, 4, 21),
    datetime.date(2000, 4, 24),
    datetime.date(2000, 5, 1),
    datetime.date(2000, 12, 25),
    datetime.date(2000, 12, 26),
    datetime.date(2001, 1, 1),
    datetime.date(2001, 4, 13),
    datetime.date(2001, 4, 16),
    datetime.date(2001, 5, 1),
    datetime.date(2001, 12, 25),
    datetime.date(2001, 12, 26),
    datetime.date(2002, 1, 1),
    datetime.date(2002, 3, 29),
    datetime.date(2002, 4, 1),
    datetime.date(2002, 5, 1),
    datetime.date(2002, 12, 25),
    datetime.date(2002, 12, 26),
    datetime.date(2003, 1, 1),
    datetime.date(2003, 4, 18),
    datetime.date(2003, 4, 21),
    datetime.date(2003, 5, 1),
    datetime.date(2003, 12, 25),
    datetime.date(2003, 12, 26),
    datetime.date(2004, 1, 1),
    datetime.date(2004, 4, 9),
    datetime.date(2004, 4, 12),
    datetime.date(2004, 5, 1),
    datetime.date(2004, 12, 25),
    datetime.date(2004, 12, 26),
    datetime.date(2005, 1, 1),
    datetime.date(2005, 3, 25),
    datetime.date(2005, 3, 28),
    datetime.date(2005, 5, 1),
    datetime.date(2005, 12, 25),
    datetime.date(2005, 12, 26),
    datetime.date(2006, 1, 1),
    datetime.date(2006, 4, 14),
    datetime.date(2006, 4, 17),
    datetime.date(2006, 5, 1),
    datetime.date(2006, 12, 25),
    datetime.date(2006, 12, 26),
    datetime.date(2007, 1, 1),
    datetime.date(2007, 4, 6),
    datetime.date(2007, 4, 9),
    datetime.date(2007, 5, 1),
    datetime.date(2007, 12, 25),
    datetime.date(2007, 12, 26),
    datetime.date(2008, 1, 1),
    datetime.date(2008, 3, 21),
    datetime.date(2008, 3, 24),
    datetime.date(2008, 5, 1),
    datetime.date(2008, 12, 25),
    datetime.date(2008, 12, 26),
    datetime.date(2009, 1, 1),
    datetime.date(2009, 4, 10),
    datetime.date(2009, 4, 13),
    datetime.date(2009, 5, 1),
    datetime.date(2009, 12, 25),
    datetime.date(2009, 12, 26),
    datetime.date(2010, 1, 1),
    datetime.date(2010, 4, 2),
    datetime.date(2010, 4, 5),
    datetime.date(2010, 5, 1),
    datetime.date(2010, 12, 25),
    datetime.date(2010, 12, 26),
    datetime.date(2011, 1, 1),
    datetime.date(2011, 4, 22),
    datetime.date(2011, 4, 25),
    datetime.date(2011, 5, 1),
    datetime.date(2011, 12, 25),
    datetime.date(2011, 12, 26),
    datetime.date(2012, 1, 1),
    datetime.date(2012, 4, 6),
    datetime.date(2012, 4, 9),
    datetime.date(2012, 5, 1),
    datetime.date(2012, 12, 25),
    datetime.date(2012, 12, 26),
    datetime.date(2013, 1, 1),
    datetime.date(2013, 3, 29),
    datetime.date(2013, 4, 1),
    datetime.date(2013, 5, 1),
    datetime.date(2013, 12, 25),
    datetime.date(2013, 12, 26),
    datetime.date(2014, 1, 1),
    datetime.date(2014, 4, 18),
    datetime.date(2014, 4, 21),
    datetime.date(2014, 5, 1),
    datetime.date(2014, 12, 25),
    datetime.date(2014, 12, 26),
    datetime.date(2015, 1, 1),
    datetime.date(2015, 4, 3),
    datetime.date(2015, 4, 6),
    datetime.date(2015, 5, 1),
    datetime.date(2015, 12, 25),
    datetime.date(2015, 12, 26),
    datetime.date(2016, 1, 1),
    datetime.date(2016, 3, 25),
    datetime.date(2016, 3, 28),
    datetime.date(2016, 5, 1),
    datetime.date(2016, 12, 25),
    datetime.date(2016, 12, 26),
    datetime.date(2017, 1, 1),
    datetime.date(2017, 4, 14),
    datetime.date(2017, 4, 17),
    datetime.date(2017, 5, 1),
    datetime.date(2017, 12, 25),
    datetime.date(2017, 12, 26),
    datetime.date(2018, 1, 1),
    datetime.date(2018, 3, 30),
    datetime.date(2018, 4, 2),
    datetime.date(2018, 5, 1),
    datetime.date(2018, 12, 25),
    datetime.date(2018, 12, 26),
    datetime.date(2019, 1, 1),
    datetime.date(2019, 4, 19),
    datetime.date(2019, 4, 22),
    datetime.date(2019, 5, 1),
    datetime.date(2019, 12, 25),
    datetime.date(2019, 12, 26),
    datetime.date(2020, 1, 1),
    datetime.date(2020, 4, 10),
    datetime.date(2020, 4, 13),
    datetime.date(2020, 5, 1),
    datetime.date(2020, 12, 25),
    datetime.date(2020, 12, 26),
    datetime.date(2021, 1, 1),
    datetime.date(2021, 4, 2),
    datetime.date(2021, 4, 5),
    datetime.date(2021, 5, 1),
    datetime.date(2021, 12, 25),
    datetime.date(2021, 12, 26),
    datetime.date(2022, 1, 1),
    datetime.date(2022, 4, 15),
    datetime.date(2022, 4, 18),
    datetime.date(2022, 5, 1),
    datetime.date(2022, 12, 25),
    datetime.date(2022, 12, 26),
    datetime.date(2023, 1, 1),
    datetime.date(2023, 4, 7),
    datetime.date(2023, 4, 10),
    datetime.date(2023, 5, 1),
    datetime.date(2023, 12, 25),
    datetime.date(2023, 12, 26),
    datetime.date(2024, 1, 1),
    datetime.date(2024, 3, 29),
    datetime.date(2024, 4, 1),
    datetime.date(2024, 5, 1),
    datetime.date(2024, 12, 25),
    datetime.date(2024, 12, 26),
]


maturity_band = [
    "O/N",
    "T/N",
    "S/N",
    "Up to 1W",
    "Up to 1M",
    "Up to 2M",
    "Up to 3M",
    "Up to 4M",
    "Up to 5M",
    "6M",
    "7M",
    "8M",
    "9M",
    "10M",
    "11M",
    "12M",
    "12M+",
    "2Y",
    "2Y-3Y",
    "3Y-5Y",
    "5Y-10Y",
    "10Y+",
    "FD",
    "NA",
]

dic_tenor = {
    "O/N": 1,
    "T/N": 1,
    "S/N": 1,
    "Up to 1W": 5,
    "Up to 1M": 20,
    "Up to 2M": 40,
    "Up to 3M": 60,
    "Up to 4M": 80,
    "Up to 5M": 100,
    "6M": 120,
    "7M": 140,
    "8M": 160,
    "9M": 180,
    "10M": 200,
    "11M": 220,
    "12M": 240,
    "12M+": 360,
    "2Y": 480,
    "2Y-3Y": 600,
    "3Y-5Y": 960,
    "5Y-10Y": 1800,
    "10Y+": 2400,
    "FD": np.nan,
    "NA": np.nan,
}
