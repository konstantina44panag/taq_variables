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
    time_col[missing] = pd.to_datetime(
        time_col[missing], format="%H:%M:%S", errors="coerce"
    ).dt.time
    return time_col.dt.time
    
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
trades = trades.rename(columns={"TIME_M": "time", "PRICE": "price", "SIZE": "vol"})
Ask = Ask.rename(columns={"TIME_M": "time", "BEST_ASK": "price", "Best_AskSizeShares": "vol"})
Bid = Bid.rename(columns={"TIME_M": "time", "BEST_BID": "price", "Best_BidSizeShares": "vol"})

trades["time"] = trades["time"].astype(str).astype(float).astype(np.float64)
Ask["time"] = Ask["time"].astype(str).astype(float).astype(np.float64)
Bid["time"] = Bid["time"].astype(str).astype(float).astype(np.float64)

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

#More datasets for analysis

Buys_trades = tradessigns[tradessigns["Initiator"] == 1][["time", "price", "vol"]]
Sells_trades = tradessigns[tradessigns["Initiator"] == -1][["time", "price", "vol"]]
tradeswithsign = tradessigns[["time", "price", "vol"]]


#Set the time index

trades["time"] = handle_time_format(trades["time"])
Ask["time"] = handle_time_format(Ask["time"])
Bid["time"] = handle_time_format(Bid["time"])
Buys_trades["time"] = handle_time_format(Buys_trades["time"])
Sells_trades["time"] = handle_time_format(Sells_trades["time"])
tradeswithsign["time"] = handle_time_format(tradeswithsign["time"])


