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
    # Print sample data to understand what you're working with
    print("Sample data before conversion:", time_col.head())

    # Convert using a specified format or try inferring it
    converted_time = pd.to_datetime(time_col, errors='coerce')
    if converted_time.isna().all():
        print("All conversions failed, attempting with another format or handling...")
        # Attempt another format or a custom handling strategy here
        # For example:
        converted_time = pd.to_datetime(time_col, format="%Y-%m-%d %H:%M:%S", errors='coerce')

    # Check results after conversion attempt
    print("Sample data after conversion:", converted_time.head())

    return converted_time

def time_to_seconds(t):
    return (t.hour * 3600 + t.minute * 60 + t.second) + t.microsecond / 1e6

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
        
# Main script execution
pd.set_option("display.max_rows", 500)

# Load datasets
with tables.open_file(args.hdf5_file_path, 'r') as hdf:
    trades = load_dataset(hdf, args.ctm_dataset_path, ["TIME_M", "PRICE", "SIZE"])
    Ask = load_dataset(hdf, args.complete_nbbo_dataset_path, ["TIME_M", "BEST_ASK", "Best_AskSizeShares"])
    Bid = load_dataset(hdf, args.complete_nbbo_dataset_path, ["TIME_M", "BEST_BID", "Best_BidSizeShares"])
    hdf.close()

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

trades['vol'] = trades['vol'].astype(str).astype(float).astype(np.int64)
Ask["vol"] = Ask["vol"].astype(str).astype(float).astype(np.int64)
Bid["vol"] = Bid["vol"].astype(str).astype(float).astype(np.int64)

trades["price"] = trades["price"].astype(str).astype(float).astype(np.float64)
Ask["price"] = Ask["price"].astype(str).astype(float).astype(np.float64)
Bid["price"] = Bid["price"].astype(str).astype(float).astype(np.float64)

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

#Set the time index
trades.set_index('time', inplace=True)
Ask.set_index('time', inplace=True)
Bid.set_index('time', inplace=True)

#Data cleaning
trades = trades.dropna(subset=["time", "price", "vol"])
Buys_trades = Buys_trades.dropna(subset=["time", "price", "vol"])
Sells_trades = Sells_trades.dropna(subset=["time", "price", "vol"])
Ask = Ask.dropna(subset=["time", "price", "vol"])
Bid = Bid.dropna(subset=["time", "price", "vol"])
tradeswithsign = tradeswithsign.dropna(subset=["time", "price", "vol"])


