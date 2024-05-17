#!/usr/bin/env python3.11
"""This script prepares the datasets to be implemented on algorithms for trade signs and estimation of variables."""
import pandas as pd
import datetime
import argparse
import numpy as np
import sys
import tables
from sign_algorithms import TradeAnalyzer

# Set up argparse to parse command line arguments
parser = argparse.ArgumentParser(description="Prepare datasets for trade sign analysis and variable estimation.")
parser.add_argument("hdf5_file_path", type=str, help="The path to the HDF5 file.")
parser.add_argument("base_date", type=str, help="The base date in YYYY-MM-DD format for time conversions.")
parser.add_argument("stock_name", type=str, help="Stock symbol.")
parser.add_argument("year", type=str, help="Year of the data.")
parser.add_argument("month", type=str, help="Month of the data.")
parser.add_argument("day", type=str, help="Day of the data.")
parser.add_argument("ctm_dataset_path", type=str, help="The dataset path within the HDF5 file for ctm data.")
parser.add_argument("complete_nbbo_dataset_path", type=str, help="The dataset path within the HDF5 file for complete nbbo data.")

args = parser.parse_args()

# Function definitions
def convert_float_to_datetime(df, float_column, base_date):
    """Converts float time values to datetime objects based on a base date."""
    midnight = pd.to_datetime(base_date + ' 00:00:00')
    
    df['timedelta'] = pd.to_timedelta(df[float_column], unit='s')
    
    df['datetime'] = midnight + df['timedelta']
    
    df.drop(columns=['timedelta'], inplace=True)
    
    return df

    
def load_dataset(hdf_file, dataset_path, columns_of_interest, corr_pattern="CORR", cond_pattern="COND"):
    """
    Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists.
    
    Parameters:
        hdf_file (tables.File): Open HDF5 file object.
        dataset_path (str): Path within the HDF5 file to the dataset.
        columns_of_interest (list): List of specific column names to extract.
        corr_pattern (str): Pattern to match for the CORR column.
        cond_pattern (str): Pattern to match for the COND column.
    
    Returns:
        pd.DataFrame: DataFrame with the specified columns and renaming applied.
    """
    try:
        dataset = hdf_file.get_node(dataset_path)
        
        column_names = [dataset._v_attrs[attr_name] for attr_name in dataset._v_attrs._f_list() if '_kind' in attr_name]
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

def load_dataset_with_exclusion(hdf_file, dataset_path, columns_of_interest, cond_pattern="COND", exclude_pattern="NBBO"):
    """
    Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists.
    
    Parameters:
        hdf_file (tables.File): Open HDF5 file object.
        dataset_path (str): Path within the HDF5 file to the dataset.
        columns_of_interest (list): List of specific column names to extract.
        cond_pattern (str): Pattern to match for the COND column.
        exclude_pattern (str): Pattern to exclude columns.
    
    Returns:
        pd.DataFrame: DataFrame with the specified columns and renaming applied.
    """
    try:
        dataset = hdf_file.get_node(dataset_path)
        
        column_names = [dataset._v_attrs[attr_name] for attr_name in dataset._v_attrs._f_list() if '_kind' in attr_name]
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

def clean_zeros(df):
    return df.dropna(subset=['price']).loc[df['price'] != 0]

# Main script execution
pd.set_option("display.max_rows", 10)

# Load datasets
with tables.open_file(args.hdf5_file_path, 'r') as hdf:
    trades = load_dataset(hdf, args.ctm_dataset_path, ["TIME_M", "PRICE", "SIZE"], corr_pattern="CORR", cond_pattern="COND")
    nbbos = load_dataset_with_exclusion(hdf, args.complete_nbbo_dataset_path, ["TIME_M", "BEST_ASK", "Best_AskSizeShares","BEST_BID", "Best_BidSizeShares"], cond_pattern="COND", exclude_pattern="NBBO")
    hdf.close()

#Data Cleaning
#Delete entries for which the spread is negative
nbbos_cleaned = nbbos[nbbos['BEST_ASK'] >= nbbos['BEST_BID']]

Ask = nbbos_cleaned[['TIME_M', 'BEST_ASK', 'Best_AskSizeShares', 'qu_cond']].copy()
Ask.rename(columns={'TIME_M': 'regular_time', 'BEST_ASK': 'price', 'Best_AskSizeShares': 'vol'}, inplace=True)
Bid = nbbos_cleaned[['TIME_M', 'BEST_BID', 'Best_BidSizeShares', 'qu_cond']].copy()
Bid.rename(columns={'TIME_M': 'regular_time', 'BEST_BID': 'price', 'Best_BidSizeShares': 'vol'}, inplace=True)

trades = trades.rename(columns={"TIME_M": "regular_time", "PRICE": "price", "SIZE": "vol"})

#Delete entries with a bid, ask or transaction price equal to zero
trades = clean_zeros(trades)
Ask = clean_zeros(Ask)
Bid = clean_zeros(Bid)

#tr_corr_columns = [col for col in trades.columns if 'TR_CORR' in col]

#for col in tr_corr_columns:
#    trades = trades[(trades[col] != 0) & (trades[col] != '00')]

#Delete entries with corrected trades
#tr_cond_columns = [col for col in trades.columns if 'TR_COND' in col]

#Delete entries with abnormal Sale Condition
#for col in tr_cond_columns:
#    trades = trades[trades[col].apply(lambda x: not x.isalpha() or x in ['E', 'F'])]


trades["regular_time"] = trades["regular_time"].astype(str).astype(float).astype(np.float64)
Ask["regular_time"] = Ask["regular_time"].astype(str).astype(float).astype(np.float64)
Bid["regular_time"] = Bid["regular_time"].astype(str).astype(float).astype(np.float64)


trades["time"] = trades["regular_time"]
Ask["time"] = Ask["regular_time"]
Bid["time"] = Bid["regular_time"]

trades = convert_float_to_datetime(trades, 'regular_time', args.base_date)
Ask = convert_float_to_datetime(Ask, 'regular_time', args.base_date)
Bid = convert_float_to_datetime(Bid, 'regular_time', args.base_date)

trades['vol'] = trades['vol'].astype(str).astype(float).astype(np.int64)
Ask["vol"] = Ask["vol"].astype(str).astype(float).astype(np.int64)
Bid["vol"] = Bid["vol"].astype(str).astype(float).astype(np.int64)

trades["price"] = trades["price"].astype(str).astype(float).astype(np.float64)
Ask["price"] = Ask["price"].astype(str).astype(float).astype(np.float64)
Bid["price"] = Bid["price"].astype(str).astype(float).astype(np.float64)

trades.reset_index(drop=True, inplace=True)
Ask.reset_index(drop=True, inplace=True)
Bid.reset_index(drop=True, inplace=True)

# Trade sign estimation
analyzer = TradeAnalyzer(trades, Ask, Bid)
tradessigns = analyzer.classify_trades()

#Datasets for analysis
Ask = Ask[["datetime", "price", "vol", "qu_cond"]].rename(columns={"datetime": "time"})
Bid = Bid[["datetime", "price", "vol", "qu_cond"]].rename(columns={"datetime": "time"})

Buys_trades = tradessigns[tradessigns["Initiator"] == 1][["datetime", "price", "vol", "corr", "cond"]].rename(columns={"datetime": "time"})
Sells_trades = tradessigns[tradessigns["Initiator"] == -1][["datetime", "price", "vol", "corr", "cond"]].rename(columns={"datetime": "time"})
trades = tradessigns[["datetime", "price", "vol", "Initiator", "corr", "cond"]].rename(columns={"datetime": "time"})

