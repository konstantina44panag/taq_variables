#!/usr/bin/env python3.11
import pandas as pd
import numpy as np
import tables
import logging
import argparse
import time
import sys
import traceback
from datetime import datetime
import polars as pl


pd.set_option('display.max_rows', 100)
parser = argparse.ArgumentParser(
    description="Prepare datasets for trade sign analysis and variable estimation."
)
parser.add_argument(
    "hdf5_file_path", type=str, help="The path to the original HDF5 file."
)
parser.add_argument(
    "ctm_dataset_path",
    type=str,
    help="The dataset path within the HDF5 file for ctm data.",
)

args, unknown = parser.parse_known_args()


    
#def load_dataset: For loading the trades dataframe from the ctm files, the necessary columns for cleaning and variables are TIME_M, PRICE, SIZE, TR_CORR, SALE_COND, in some files the names change to TRCORR, TRCOND, 
#so I identify the columns by the patterns CORR, COND
#The data are loaded to the trades dataframe
def load_trades_dataset(
    hdf_file,
    dataset_path,
    suf_pattern="SUF",
):
    """Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists."""
    try:
        table_path=f"{dataset_path}table"
        dataset = hdf_file.get_node(table_path)

        column_names =  hdf_file.get_node(dataset_path)._v_attrs["column_names"]
        data = {}

        for col in column_names:
            if suf_pattern in col:
                data["suffix"] = dataset.col(col)
        df = pd.DataFrame(data)
        return df

    except tables.NoSuchNodeError as e:
        return None
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
        

#def decode_byte_strings: The data in HDF5 files have the type byte strings, I decode the data to string type    
def decode_byte_strings(df):
    """Decode byte strings in all columns of the dataframe."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    return df

def get_unique_suffix_values(df):
    """Get unique values in the 'suffix' column, including nan as a separate category."""
    # Convert the column to a NumPy array and replace NaNs with 'nan'
    suffix_array = df["suffix"].to_numpy()
    suffix_array = np.where(pd.isna(suffix_array), "nan", suffix_array)
    
    # Get unique values
    unique_values = np.unique(suffix_array)
    
    return unique_values


with tables.open_file(args.hdf5_file_path, "r") as hdf:
    trades = load_trades_dataset(
        hdf,
        args.ctm_dataset_path,
        suf_pattern="SUF",
    )
if trades is None:
    print("None")
else:
    trades = decode_byte_strings(trades)
    trades["suffix"] = trades["suffix"].astype(str)
    unique_suffix_values = get_unique_suffix_values(trades)
    print(",".join(unique_suffix_values))