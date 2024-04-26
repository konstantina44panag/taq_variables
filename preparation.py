#!/usr/bin/env python3.11
"""This script prepares the datasets to be implemented on algorithms for trade signs and estimation of variables."""
import pandas as pd
import argparse
import numpy as np
import sys
import tables
from sign_algorithms import TradeAnalyzer

# Set up argparse to parse command line arguments
parser = argparse.ArgumentParser(
    description="Prepare datasets for trade sign analysis and variable estimation."
)
parser.add_argument("hdf5_file_path", type=str, help="The path to the HDF5 file.")
parser.add_argument(
    "ctm_dataset_path",
    type=str,
    help="The dataset path within the HDF5 file for ctm data."
)
parser.add_argument(
    "complete_nbbo_dataset_path",
    type=str,
    help="The dataset path within the HDF5 file for complete nbbo data."
)
args = parser.parse_args()

# Function definitions
def handle_time_format(time_col):
    time_col = pd.to_datetime(time_col, format="%H:%M:%S.%f", errors="coerce")
    missing = time_col.isna()
    time_col[missing] = pd.to_datetime(time_col[missing], format="%H:%M:%S", errors="coerce").dt.time
    return time_col.dt.time

def time_to_seconds(t):
    return (t.hour * 3600 + t.minute * 60 + t.second) + t.microsecond / 1e6

def has_16_or_fewer_characters(x):
    """
    Check if the string has 16 or fewer characters and handle appropriately.
    If more than 16 characters are found, truncate to 16 characters and log a message.
    """
    try:
        str_x = str(x)  
        if len(str_x) > 16:
            sys.stderr.write(f"Warning: Value longer than 16 digits detected; truncated to 16 digits: {str_x[:16]}\n")
            return str_x[:16] 
    except Exception as e:
        sys.stderr.write(f"Error processing the value {x}: {e}\n")
    return x

def load_dataset(hdf_file, dataset_path, columns_of_interest):
    """Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists."""
    try:
        dataset = hdf_file.get_node(dataset_path)
        
        column_names = [dataset._v_attrs[attr_name] for attr_name in dataset._v_attrs._f_list() if '_kind' in attr_name]
        column_names = [item for sublist in column_names for item in sublist]  # Flatten list if needed

        data = {}
        for col in column_names:
            if col in columns_of_interest:
                data[col] = dataset.col(col)

        return pd.DataFrame(data)

    except tables.NoSuchNodeError as e:
        raise ValueError(f"Dataset path not found: {dataset_path}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
        
def apply_column_checks(df, columns):
    """Apply checks and conversions on dataframe columns."""
    for column in columns:
        if df[column].dtype == 'O':  # Assuming 'O' type means it's a string
            df[column] = df[column].apply(has_16_or_fewer_characters)
    return df

# Main script execution
pd.set_option("display.max_rows", 500)

# Load datasets
with tables.open_file(args.hdf5_file_path, 'r') as hdf:
    trades = load_dataset(hdf, args.ctm_dataset_path, ["TIME_M", "PRICE", "SIZE"])
    Ask = load_dataset(hdf, args.complete_nbbo_dataset_path, ["TIME_M", "BEST_ASK", "Best_AskSizeShares"])
    Bid = load_dataset(hdf, args.complete_nbbo_dataset_path, ["TIME_M", "BEST_BID", "Best_BidSizeShares"])
    hdf.close()

trades = apply_column_checks(trades, ["PRICE", "SIZE"])
Ask = apply_column_checks(Ask, ["BEST_ASK", "Best_AskSizeShares"])
Bid = apply_column_checks(Bid, ["BEST_BID", "Best_BidSizeShares"])

# Rename and convert columns
trades = trades.rename(columns={"TIME_M": "regular_time", "PRICE": "price", "SIZE": "vol"})
Ask = Ask.rename(columns={"TIME_M": "regular_time", "BEST_ASK": "price", "Best_AskSizeShares": "vol"})
Bid = Bid.rename(columns={"TIME_M": "regular_time", "BEST_BID": "price", "Best_BidSizeShares": "vol"})

trades["regular_time"] = handle_time_format(trades["regular_time"])
Ask["regular_time"] = handle_time_format(Ask["regular_time"])
Bid["regular_time"] = handle_time_format(Bid["regular_time"])

trades.reset_index(drop=True, inplace=True)
Ask.reset_index(drop=True, inplace=True)
Bid.reset_index(drop=True, inplace=True)

# Convert time and adjust data types
trades["time"] = trades["regular_time"].apply(time_to_seconds)
Ask["time"] = Ask["regular_time"].apply(time_to_seconds)
Bid["time"] = Bid["regular_time"].apply(time_to_seconds)

trades["vol"] = trades["vol"].astype(np.int64)
Ask["vol"] = Ask["vol"].astype(np.int64)
Bid["vol"] = Bid["vol"].astype(np.int64)

trades["price"] = trades["price"].astype(np.float64)
Ask["price"] = Ask["price"].astype(np.float64)
Bid["price"] = Bid["price"].astype(np.float64)

# Trade sign estimation
analyzer = TradeAnalyzer(trades, Ask, Bid)
tradessigns = analyzer.classify_trades()

print(tradessigns)

#Datasets for analysis
trades = trades[["regular_time", "price", "vol"]].rename(columns={"regular_time": "time"})
Ask = Ask[["regular_time", "price", "vol"]].rename(columns={"regular_time": "time"})
Bid = Bid[["regular_time", "price", "vol"]].rename(columns={"regular_time": "time"})

Buys_trades = tradessigns[tradessigns["Initiator"] == 1][
    ["regular_time", "price", "vol"]
].rename(columns={"regular_time": "time"})
Sells_trades = tradessigns[tradessigns["Initiator"] == -1][
    ["regular_time", "price", "vol"]
].rename(columns={"regular_time": "time"})
tradeswithsign = tradessigns[["regular_time", "price", "vol"]].rename(
    columns={"regular_time": "time"}
)


Buys_trades["time"] = handle_time_format(Buys_trades["time"])
Sells_trades["time"] = handle_time_format(Sells_trades["time"])
tradeswithsign["time"] = handle_time_format(tradeswithsign["time"])

#Data cleaning
trades = trades.dropna(subset=["time", "price", "vol"])
Buys_trades = Buys_trades.dropna(subset=["time", "price", "vol"])
Sells_trades = Sells_trades.dropna(subset=["time", "price", "vol"])
Ask = Ask.dropna(subset=["time", "price", "vol"])
Bid = Bid.dropna(subset=["time", "price", "vol"])
tradeswithsign = tradeswithsign.dropna(subset=["time", "price", "vol"])


