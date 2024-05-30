#!/usr/bin/env python3.11
"""This script prepares the datasets to be implemented on algorithms for trade signs and estimation of variables."""
import pandas as pd
import argparse
import numpy as np
import tables
from sign_algorithms import TradeAnalyzer
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up argparse to parse command line arguments
parser = argparse.ArgumentParser(
    description="Prepare datasets for trade sign analysis and variable estimation."
)
parser.add_argument("hdf5_file_path", type=str, help="The path to the HDF5 file.")
parser.add_argument(
    "base_date",
    type=str,
    help="The base date in YYYY-MM-DD format for time conversions.",
)
parser.add_argument("stock_name", type=str, help="Stock symbol.")
parser.add_argument("year", type=str, help="Year of the data.")
parser.add_argument("month", type=str, help="Month of the data.")
parser.add_argument("day", type=str, help="Day of the data.")
parser.add_argument(
    "ctm_dataset_path",
    type=str,
    help="The dataset path within the HDF5 file for ctm data.",
)
parser.add_argument(
    "complete_nbbo_dataset_path",
    type=str,
    help="The dataset path within the HDF5 file for complete nbbo data.",
)
args = parser.parse_args()


# Function definitions
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


def handle_duplicates(df, price_col, other_cols, datetime_col="datetime"):
    """Handle duplicates by taking the median for the price column and the last value for other columns."""
    agg_funcs = {price_col: "median"}
    for col in other_cols:
        agg_funcs[col] = "last"
    agg_funcs[datetime_col] = "last"
    df = df.groupby("time").agg(agg_funcs).reset_index()
    return df


def clean_zeros(df):
    return df.dropna(subset=["price"]).loc[df["price"] != 0]


# Main script execution
pd.set_option("display.max_rows", 10)

try:
    # Load datasets
    with tables.open_file(args.hdf5_file_path, "r") as hdf:
        trades = load_dataset(
            hdf,
            args.ctm_dataset_path,
            ["TIME_M", "PRICE", "SIZE"],
            corr_pattern="CORR",
            cond_pattern="COND",
        )
        nbbos = load_dataset_with_exclusion(
            hdf,
            args.complete_nbbo_dataset_path,
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

    trades["TIME_M"] = trades["TIME_M"].astype(str).astype(float).astype(np.float64)
    nbbos["TIME_M"] = nbbos["TIME_M"].astype(str).astype(float).astype(np.float64)

    trades["time"] = trades["TIME_M"]
    nbbos["time"] = nbbos["TIME_M"]

    trades = convert_float_to_datetime(trades, "TIME_M", args.base_date)
    nbbos = convert_float_to_datetime(nbbos, "TIME_M", args.base_date)

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
    nbbos_cleaned = nbbos[nbbos["BEST_ASK"] >= nbbos["BEST_BID"]]
    trades = trades[trades['corr'] == '00']

    Ask = nbbos_cleaned[
        ["time", "datetime", "BEST_ASK", "Best_AskSizeShares", "qu_cond"]
    ].copy()
    Ask.rename(columns={"BEST_ASK": "price", "Best_AskSizeShares": "vol"}, inplace=True)
    Bid = nbbos_cleaned[
        ["time", "datetime", "BEST_BID", "Best_BidSizeShares", "qu_cond"]
    ].copy()
    Bid.rename(columns={"BEST_BID": "price", "Best_BidSizeShares": "vol"}, inplace=True)
    trades = trades.rename(columns={"PRICE": "price", "SIZE": "vol"})

    Ask = handle_duplicates(Ask, "price", ["vol", "qu_cond"])
    Bid = handle_duplicates(Bid, "price", ["vol", "qu_cond"])
    trades = handle_duplicates(trades, "price", ["vol", "corr", "cond"])

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

    assert not trades["vol"].isna().any(), "NA values found in trades['vol']"
    assert not Ask["vol"].isna().any(), "NA values found in Ask['vol']"
    assert not Bid["vol"].isna().any(), "NA values found in Bid['vol']"
    assert not np.isinf(trades["vol"]).any(), "Inf values found in trades['vol']"
    assert not np.isinf(Ask["vol"]).any(), "Inf values found in Ask['vol']"
    assert not np.isinf(Bid["vol"]).any(), "Inf values found in Bid['vol']"

    trades.reset_index(drop=True, inplace=True)
    Ask.reset_index(drop=True, inplace=True)
    Bid.reset_index(drop=True, inplace=True)

    logging.info("Estimating trade signs")
    analyzer = TradeAnalyzer(trades, Ask, Bid)
    tradessigns = analyzer.classify_trades()

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
        ["datetime", "price", "vol", "corr", "cond"]
    ].rename(columns={"datetime": "time"})
    Sells_trades = tradessigns[tradessigns["Initiator"] == -1][
        ["datetime", "price", "vol", "corr", "cond"]
    ].rename(columns={"datetime": "time"})
    trades_v1 = tradessigns[
        ["datetime", "price", "vol", "Initiator", "corr", "cond"]
    ].rename(columns={"datetime": "time"})
    trades_v2 = pd.merge(
        trades[["datetime", "price", "vol", "corr", "cond"]].rename(
            columns={"datetime": "time"}
        ),
        tradessigns[["datetime", "Initiator"]].rename(columns={"datetime": "time"}),
        on="time",
        how="inner",
    )

    trades_v2 = trades_v2[trades_v1.columns]
    if trades_v1.equals(trades_v2):
        logging.info("The two DataFrames are the same.")
    else:
        logging.error("The two DataFrames are not the same.")
        if trades_v1.shape != trades_v2.shape:
            logging.error(
                f"Shape mismatch: trades_v1.shape={trades_v1.shape}, trades_v2.shape={trades_v2.shape}"
            )
        if list(trades_v1.columns) != list(trades_v2.columns):
            logging.error(
                f"Column names mismatch: trades_v1.columns={trades_v1.columns}, trades_v2.columns={trades_v2.columns}"
            )
        comparison = trades_v1.compare(trades_v2, keep_shape=True, keep_equal=True)
        logging.error(f"Differences:\n{comparison}")
        sys.exit(1)

    trades = trades_v1
    Buys_trades.reset_index(drop=True, inplace=True)
    Sells_trades.reset_index(drop=True, inplace=True)
    trades.sort_values(by="time", inplace=True)
    Ask.sort_values(by="time", inplace=True)
    Bid.sort_values(by="time", inplace=True)
    Buys_trades.sort_values(by="time", inplace=True)
    Sells_trades.sort_values(by="time", inplace=True)


except Exception as e:
    logging.error(f"An error occurred: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)
