import pandas as pd
import numpy as np
import tables
from sign_algorithms import TradeAnalyzer
import logging
import time
import sys
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function definitions
def print_debug_info(df, name):
    print(f"\n{name}")
    print(df.head())
    print(df.columns)
    print(f"DataFrame shape: {df.shape}")
    
def convert_float_to_datetime(df, float_column, base_date):
    """Converts float time values to datetime objects based on a base date."""
    midnight = pd.to_datetime(base_date + " 00:00:00")
    df["timedelta"] = pd.to_timedelta(df[float_column], unit="s")
    df["datetime"] = midnight + df["timedelta"]
    df.drop(columns=["timedelta"], inplace=True)
    return df

def load_dataset(
    hdf_file,
    dataset_path,
    columns_of_interest,
    corr_pattern="CORR",
    cond_pattern="COND",
):
    """Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists."""
    try:
        dataset = hdf_file.get_node(dataset_path)
        column_names = [
            dataset._v_attrs[attr_name]
            for attr_name in dataset._v_attrs._f_list()
            if "_kind" in attr_name
        ]
        column_names = [item for sublist in column_names for item in sublist]

        data = {}
        for col in columns_of_interest:
            if col in column_names:
                data[col] = dataset.col(col)

        for col in column_names:
            if corr_pattern in col and "corr_col" not in data:
                data["corr"] = dataset.col(col)
            elif cond_pattern in col and "cond_col" not in data:
                data["cond"] = dataset.col(col)

        df = pd.DataFrame(data)
        return df

    except tables.NoSuchNodeError as e:
        raise ValueError(f"Dataset path not found: {dataset_path}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def load_dataset_with_exclusion(
    hdf_file,
    dataset_path,
    columns_of_interest,
    cond_pattern="COND",
    exclude_pattern="NBBO",
):
    """Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists."""
    try:
        dataset = hdf_file.get_node(dataset_path)
        column_names = [
            dataset._v_attrs[attr_name]
            for attr_name in dataset._v_attrs._f_list()
            if "_kind" in attr_name
        ]
        column_names = [item for sublist in column_names for item in sublist]

        data = {}
        for col in columns_of_interest:
            if col in column_names:
                data[col] = dataset.col(col)

        for col in column_names:
            if cond_pattern in col and exclude_pattern not in col:
                data["qu_cond"] = dataset.col(col)

        df = pd.DataFrame(data)
        return df

    except tables.NoSuchNodeError as e:
        raise ValueError(f"Dataset path not found: {dataset_path}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def decode_byte_strings(df):
    """Decode byte strings in all columns of the dataframe."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    return df

def find_na_or_inf(df):
    na_mask = df.isna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols])
    combined_mask = na_mask | inf_mask
    na_inf_rows = df[combined_mask.any(axis=1)]
    return na_inf_rows

def handle_duplicates(df, key_col, value_cols, other_cols=None):
    """
    Handle duplicates by taking the median for the value columns and the last value for other columns.
    If other_cols is not provided, only the value_cols will be aggregated.
    """
    agg_funcs = {col: "median" for col in value_cols}
    if other_cols:
        for col in other_cols:
            agg_funcs[col] = "last"
    df = df.groupby(key_col).agg(agg_funcs).reset_index()
    return df

def clean_zeros(df):
    return df.dropna(subset=["price"]).loc[df["price"] != 0]

def identify_retail(z):
            if 0 < z < 0.4 or 0.6 < z < 1:
                return 'retail trade'
            else:
                return 'non-retail trade'
            
def calculate_returns(df, price_col='price', time_col='time'):
    """
    Calculate returns from a DataFrame containing price and time columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing price and time columns.
    price_col (str): The name of the price column. Default is 'price'.
    time_col (str): The name of the time column. Default is 'time'.

    Returns:
    pd.DataFrame: A DataFrame containing the time and returns columns.
    """
    df["log_price"] = np.log(df[price_col])
    
    returns_df = pd.DataFrame()
    returns_df["time"] = df[time_col]
    returns_df["returns"] = df["log_price"].diff()
    
    returns_df = returns_df.dropna().reset_index(drop=True)
    
    df.drop(columns=["log_price"], inplace=True)
    
    return returns_df

def calculate_returns_shift(df, price_col='price', time_col='time', additional_cols=[]):
    """
    Calculate returns from a DataFrame containing price and time columns using the shift method.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing price and time columns.
    price_col (str): The name of the price column. Default is 'price'.
    time_col (str): The name of the time column. Default is 'time'.
    additional_cols (list): List of additional columns to keep in the output DataFrame. Default is an empty list.

    Returns:
    pd.DataFrame: A DataFrame containing the time, returns, and additional columns.
    """
    returns_df = pd.DataFrame()
    returns_df["time"] = df[time_col]
    returns_df["returns"] = np.log(df[price_col] / df[price_col].shift(1))
    
    if additional_cols:
        for col in additional_cols:
            returns_df[col] = df[col]

    returns_df = returns_df.reset_index(drop=True)
    
    return returns_df

def prepare_datasets(hdf5_file_path, base_date, stock_name, year, month, day, ctm_dataset_path, complete_nbbo_dataset_path):
    try:
        load_start_time = time.time()

        # Load datasets
        with tables.open_file(hdf5_file_path, "r") as hdf:
            trades = load_dataset(
                hdf,
                ctm_dataset_path,
                ["TIME_M", "EX", "PRICE", "SIZE"],
                corr_pattern="CORR",
                cond_pattern="COND",
            )
            nbbos = load_dataset_with_exclusion(
                hdf,
                complete_nbbo_dataset_path,
                [
                    "TIME_M",
                    "BEST_ASK",
                    "Best_AskSizeShares",
                    "BEST_BID",
                    "Best_BidSizeShares",
                ],
                cond_pattern="COND",
                exclude_pattern="NBBO",
            )
            hdf.close()

        load_end_time = time.time()
        load_time = load_end_time - load_start_time

        clean_start_time = time.time()

        trades = decode_byte_strings(trades)
        nbbos = decode_byte_strings(nbbos)

        logging.info("Cleaning data")
        trades["PRICE"] = pd.to_numeric(trades["PRICE"], errors="coerce")
        trades["SIZE"] = pd.to_numeric(trades["SIZE"], errors="coerce")
       
        trades = trades.dropna(subset=["PRICE", "SIZE"]).loc[
            (trades["PRICE"] != 0) & (trades["SIZE"] != 0)
        ]
        columns_to_convert = [
            "Best_AskSizeShares",
            "Best_BidSizeShares",
            "BEST_ASK",
            "BEST_BID",
        ]
        for col in columns_to_convert:
            nbbos[col] = pd.to_numeric(nbbos[col], errors="coerce")
        nbbos = nbbos.dropna(
            subset=["Best_AskSizeShares", "Best_BidSizeShares", "BEST_ASK", "BEST_BID"]
        ).loc[
            (nbbos["Best_AskSizeShares"] != 0)
            & (nbbos["Best_BidSizeShares"] != 0)
            & (nbbos["BEST_ASK"] != 0)
            & (nbbos["BEST_BID"] != 0)
        ]
        #Formatting
        trades["EX"] = trades["EX"].astype(str)

        trades["TIME_M"] = trades["TIME_M"].astype(str).astype(float).astype(np.float64)
        nbbos["TIME_M"] = nbbos["TIME_M"].astype(str).astype(float).astype(np.float64)
        trades["time"] = trades["TIME_M"]
        nbbos["time"] = nbbos["TIME_M"]
        trades.drop(columns=["TIME_M"], inplace=True)
        nbbos.drop(columns=["TIME_M"], inplace=True)

        trades["SIZE"] = trades["SIZE"].astype(float).astype(np.int64)
        nbbos["Best_AskSizeShares"] = (
            nbbos["Best_AskSizeShares"].astype(float).astype(np.int64)
        )
        nbbos["Best_BidSizeShares"] = (
            nbbos["Best_BidSizeShares"].astype(float).astype(np.int64)
        )

        trades["PRICE"] = trades["PRICE"].astype(float).astype(np.float64)
        nbbos["BEST_ASK"] = nbbos["BEST_ASK"].astype(float).astype(np.float64)
        nbbos["BEST_BID"] = nbbos["BEST_BID"].astype(float).astype(np.float64)

        # Data Cleaning
        trades = trades[trades['corr'] == '00']
        trades = trades.rename(columns={"PRICE": "price", "SIZE": "vol"})

        nbbos = handle_duplicates(nbbos, key_col='time', value_cols=['BEST_ASK', 'BEST_BID'], other_cols=['Best_AskSizeShares', 'Best_BidSizeShares', 'qu_cond'])
        trades = handle_duplicates(trades, key_col='time', value_cols=['price'], other_cols=['vol', "corr", "cond", "EX"])

        nbbos = nbbos[nbbos["BEST_ASK"] >= nbbos["BEST_BID"]]

        trades = convert_float_to_datetime(trades, "time", base_date)
        nbbos = convert_float_to_datetime(nbbos, "time", base_date)

        #Define Ask, Bid, Midprice dataframes
        Ask = nbbos[
            ["time", "datetime", "BEST_ASK", "Best_AskSizeShares", "qu_cond"]
        ].copy()
        Ask.rename(columns={"BEST_ASK": "price", "Best_AskSizeShares": "vol"}, inplace=True)
        Bid = nbbos[
            ["time", "datetime", "BEST_BID", "Best_BidSizeShares", "qu_cond"]
        ].copy()
        Bid.rename(columns={"BEST_BID": "price", "Best_BidSizeShares": "vol"}, inplace=True)
        nbbos['midpoint'] = (nbbos['BEST_BID'] + nbbos['BEST_ASK']) / 2
        Midpoint = nbbos[['datetime', 'midpoint']].rename(
            columns={"datetime" : "time", "midpoint": "price"}
        )
        nbbos['bid_change'] = nbbos['BEST_BID'].diff().fillna(0) != 0
        nbbos['bid_size_change'] = nbbos['Best_BidSizeShares'].diff().fillna(0) != 0
        nbbos['ask_change'] = nbbos['BEST_ASK'].diff().fillna(0) != 0
        nbbos['ask_size_change'] = nbbos['Best_AskSizeShares'].diff().fillna(0) != 0

        nbbos['sign'] = ((nbbos['bid_change'] | nbbos['bid_size_change']).astype(int) - 
                 (nbbos['ask_change'] | nbbos['ask_size_change']).astype(int))
        nbbo_signs = nbbos[['datetime', 'sign']].rename(
            columns={"datetime": "time"}
        )

        logging.info("Checking for NA or inf values before conversion to int")
        na_inf_trades = find_na_or_inf(trades)
        na_inf_ask = find_na_or_inf(Ask)
        na_inf_bid = find_na_or_inf(Bid)

        if not na_inf_trades.empty:
            logging.warning("NA or inf values in trades dataframe:")
            logging.warning(na_inf_trades)

        if not na_inf_ask.empty:
            logging.warning("NA or inf values in Ask dataframe:")
            logging.warning(na_inf_ask)

        if not na_inf_bid.empty:
            logging.warning("NA or inf values in Bid dataframe:")
            logging.warning(na_inf_bid)

        trades.reset_index(drop=True, inplace=True)
        Ask.reset_index(drop=True, inplace=True)
        Bid.reset_index(drop=True, inplace=True)
        Midpoint.reset_index(drop=True, inplace=True)
        nbbo_signs.reset_index(drop=True, inplace=True)

        trsigns_start_time = time.time()
        logging.info("Estimating trade signs")
        analyzer = TradeAnalyzer(trades, Ask, Bid)
        tradessigns = analyzer.classify_trades()
        trsigns_end_time = time.time()
        trsigns_time = trsigns_end_time - trsigns_start_time

        trades.reset_index(drop=True, inplace=True)
        tradessigns.reset_index(drop=True, inplace=True)
        Ask.reset_index(drop=True, inplace=True)
        Bid.reset_index(drop=True, inplace=True)

        logging.info("Preparing datasets for analysis")
        Ask = Ask[["datetime", "price", "vol", "qu_cond"]].rename(
            columns={"datetime": "time"}
        )
        Bid = Bid[["datetime", "price", "vol", "qu_cond"]].rename(
            columns={"datetime": "time"}
        )

        Buys_trades = tradessigns[tradessigns["Initiator"] == 1][
            ["datetime", "price", "vol", "corr", "cond", "EX"]
        ].rename(columns={"datetime": "time"})

        Sells_trades = tradessigns[tradessigns["Initiator"] == -1][
            ["datetime", "price", "vol", "corr", "cond", "EX"]
        ].rename(columns={"datetime": "time"})

        Retail_trades = tradessigns[tradessigns["EX"] == "D"][
            ["datetime", "price", "vol", "corr", "cond", "EX", "Initiator"]].rename(columns={"datetime": "time", "Initiator": "sign"})
        Retail_trades['Z'] = 100 * (Retail_trades['price'] % 0.01)          
        Retail_trades['trade_type'] = Retail_trades['Z'].apply(identify_retail)
        Retail_trades = Retail_trades[Retail_trades['trade_type'] == 'retail trade'].drop(columns=['trade_type'])

        target_date = datetime(2014, 1, 1)
        Oddlot_trades = tradessigns[(tradessigns['datetime'] >= target_date) & (tradessigns['cond'] == "I")][["datetime", "price", "vol", "corr", "cond", "EX", "Initiator"]].rename(columns={"datetime": "time", "Initiator": "sign"})
       
        trade_returns = calculate_returns_shift(tradessigns, price_col='price', time_col='datetime', additional_cols=['vol'])
        midprice_returns = calculate_returns_shift(Midpoint, price_col='price', time_col='time')
        print_debug_info(trade_returns, "Trade Returns:")
        print_debug_info(midprice_returns, "Midprice Returns:")
        trade_signs = tradessigns[
            ["datetime", "Initiator"]].rename(columns={"datetime": "time", "Initiator": "sign"})

        #Test that trade sign aglorithm doesnt distort the rest of the columns
        trades_test1 = tradessigns[
            ["datetime", "price", "vol", "Initiator", "corr", "cond", "EX"]
        ].rename(columns={"datetime": "time"})

        trades_test2 = pd.merge(
            trades[["datetime", "price", "vol", "corr", "cond", "EX"]].rename(
                columns={"datetime": "time"}
            ),
            tradessigns[["datetime", "Initiator"]].rename(columns={"datetime": "time"}),
            on="time",
            how="inner",
        )

        trades_test2 = trades_test2[trades_test1.columns]
        if trades_test1.equals(trades_test2):
            logging.info("The two DataFrames are the same.")
        else:
            logging.error("The two DataFrames are not the same.")
            if trades_test1.shape != trades_test2.shape:
                logging.error(
                    f"Shape mismatch: trades_test1.shape={trades_test1.shape}, trades_v2.shape={trades_test2.shape}"
                )
            if list(trades_test1.columns) != list(trades_test2.columns):
                logging.error(
                    f"Column names mismatch: trades_v1.columns={trades_test1.columns}, trades_test2.columns={trades_test2.columns}"
                )
            comparison = trades_test1.compare(trades_test2, keep_shape=True, keep_equal=True)
            logging.error(f"Differences:\n{comparison}")
            sys.exit(1)

        trades = trades_test1
        Buys_trades.reset_index(drop=True, inplace=True)
        Sells_trades.reset_index(drop=True, inplace=True)
        Retail_trades.reset_index(drop=True, inplace=True)
        trades.sort_values(by="time", inplace=True)
        Ask.sort_values(by="time", inplace=True)
        Bid.sort_values(by="time", inplace=True)
        Buys_trades.sort_values(by="time", inplace=True)
        Sells_trades.sort_values(by="time", inplace=True)
        Retail_trades.sort_values(by="time", inplace=True)
        Midpoint.sort_values(by="time", inplace=True)
        Oddlot_trades.sort_values(by="time", inplace=True)
        trade_returns.sort_values(by="time", inplace=True)
        midprice_returns.sort_values(by="time", inplace=True)
        trade_signs.sort_values(by="time", inplace=True)
        nbbo_signs.sort_values(by="time", inplace=True)


        clean_end_time = time.time()
        clean_time = clean_end_time - clean_start_time - trsigns_time
        with open("preparation_timeanalysis.txt", "a") as f:
            f.write(f"Stock: {stock_name}\n")
            f.write(f"Read time: {load_time} seconds\n")
            f.write(f"Clean time: {clean_time} seconds\n")
            f.write(f"TradeSigns time: {trsigns_time} seconds\n")

        return trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
