#!/usr/bin/env python3.11
"""This script computes important variables but first passes arguments to preparation.py."""

import pandas as pd
import numpy as np
import subprocess
import argparse

# Parse arguments
parser = argparse.ArgumentParser(
    description="Prepare datasets for trade sign analysis and variable estimation."
)
parser.add_argument(
    "hdf5_file_path", type=str, help="The path to the original HDF5 file."
)
parser.add_argument("base_date", type=str, help="Base date for the analysis.")
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

args, unknown = parser.parse_known_args()

# Constructing file paths based on the arguments
hdf5_variable_path = f"/home/taq/taq_allstocks/{args.year}{args.month}02_var_allstocks_v3_again.h5"
print(f"Output HDF5 file path: {hdf5_variable_path}")

# Run preparation.py with the required arguments
result = subprocess.run(
    [
        "python3.11",
        "preparation.py",
        args.hdf5_file_path,
        args.base_date,
        args.stock_name,
        args.year,
        args.month,
        args.day,
        args.ctm_dataset_path,
        args.complete_nbbo_dataset_path,
    ],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("Error running preparation.py:", result.stderr)
else:
    print("preparation.py ran successfully:", result.stdout)

# Import variables from preparation.py
try:
    from preparation import trades, Buys_trades, Sells_trades, Ask, Bid
except ImportError as e:
    print(f"Failed to import variables from preparation.py: {e}")
    exit(1)

# Extra Columns Necessary
Buys_trades["value"] = Buys_trades["price"] * Buys_trades["vol"]
Sells_trades["value"] = Sells_trades["price"] * Sells_trades["vol"]

# Custom aggregation functions
def last_aggr_price(data):
    filtered_trades = data[data["value"] >= 10000]
    if not filtered_trades.empty:
        return filtered_trades.iloc[-1]["price"]
    else:
        return np.nan

def calculate_vwap(data):
    weighted_prices = (data["price"] * data["vol"]).sum()
    total_volume = data["vol"].sum()
    return weighted_prices / total_volume if total_volume != 0 else 0

def calculate_twap_and_volume(group):
    durations = (
        group.index.to_series()
        .diff()
        .fillna(pd.Timedelta(seconds=0))
        .dt.total_seconds()
    )
    time_weighted_prices = group["price"] * durations
    time_weighted_volumes = group["vol"] * durations
    total_time = durations.sum()
    twap = time_weighted_prices.sum() / total_time if total_time != 0 else 0
    twav = time_weighted_volumes.sum() / total_time if total_time != 0 else 0

    return pd.Series({"TWAP": twap, "TWAV": twav})

def count_changes(series):
    price_changes = series.diff()
    num_changes = (price_changes != 0).sum()
    return num_changes

def apply_aggregations(df):
    # Ensure necessary columns exist
    if 'value' not in df.columns:
        df['value'] = df['price'] * df['vol']

    # Ensure the 'time' column exists and is set to datetime
    if 'time' not in df.columns:
        raise KeyError("'time' column is missing from the DataFrame")
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # Set 'time' as the index
    df.set_index('time', inplace=True)

    # Debug: print first few rows to check if 'price' and 'vol' columns are present
    print(f"Applying aggregations on DataFrame with columns: {df.columns.tolist()}")
    print(df.head())

    # Resample and apply custom aggregation functions
    def safe_aggregation(group):
        if group.empty or 'price' not in group or 'vol' not in group:
            return pd.Series({
                'last_price': np.nan,
                'mean_price': np.nan,
                'last_vol': np.nan,
                'sum_vol': np.nan,
                'last_time': np.nan,
                'last_aggr_price': np.nan,
                'VWAP': np.nan,
                'price_changes': np.nan,
                'vol_changes': np.nan,
                'TWAP': np.nan,
                'TWAV': np.nan
            })
        return pd.Series({
            'last_price': group['price'].iloc[-1],
            'mean_price': group['price'].mean(),
            'last_vol': group['vol'].iloc[-1],
            'sum_vol': group['vol'].sum(),
            'last_time': group.index[-1],
            'last_aggr_price': last_aggr_price(group),
            'VWAP': calculate_vwap(group),
            'price_changes': count_changes(group['price']),
            'vol_changes': count_changes(group['vol']),
            'TWAP': calculate_twap_and_volume(group)['TWAP'],
            'TWAV': calculate_twap_and_volume(group)['TWAV']
        })

    resampled = df.resample("1min").apply(safe_aggregation)

    return resampled

# Apply aggregations
dataframes_to_process = {
    "trades": trades,
    "Ask": Ask,
    "Bid": Bid,
    "Buys_trades": Buys_trades,
    "Sells_trades": Sells_trades
}

aggregated_data = {}

for name, df in dataframes_to_process.items():
    if not df.empty:
        try:
            print(f"Processing {name} DataFrame")
            agg_df = apply_aggregations(df)
        except KeyError as e:
            print(f"Error processing {name}: {e}")
            continue
        agg_df.rename(columns={
            "last_price": f"last_{name.upper()}_PRICE",
            "last_vol": f"last_{name.upper()}_VOL",
            "last_time": f"last_{name.upper()}_TIME",
            "mean_price": f"av_PRICE_{name.upper()}",
            "last_aggr_price": f"last_aggr_{name.upper()}_PRICE",
            "VWAP": f"VWAP_{name.upper()}",
            "sum_vol": f"t_VOL_{name.upper()}",
            "price_changes": f"no_dP_{name.upper()}",
            "vol_changes": f"no_dV_{name.upper()}",
            "TWAP": f"TWAP_{name.upper()}",
            "TWAV": f"TWAV_{name.upper()}",
        }, inplace=True)
        aggregated_data[name] = agg_df.between_time("09:30", "16:00")

# 7. Volume weighted average pre/post-trade bid/ask prices (measured just before each trade and weighted by the size of each trade)
# backward merger
merged_trades = pd.merge_asof(
    trades, Ask, on="time", direction="backward", suffixes=("", "_ask")
)
merged_trades = pd.merge_asof(
    merged_trades, Bid, on="time", direction="backward", suffixes=("", "_bid")
)

merged_trades["weighted_bid"] = merged_trades["price_bid"] * merged_trades["vol"]
merged_trades["weighted_ask"] = merged_trades["price_ask"] * merged_trades["vol"]
merged_trades.set_index("time", inplace=True)
merged_trades.index = pd.to_datetime(merged_trades.index)

aggregation_rules = {
    "price": "mean",
    "vol": "sum",
    "weighted_bid": "sum",
    "weighted_ask": "sum",
    "price_bid": "mean",
    "price_ask": "mean",
}

one_minute_bins = merged_trades.resample("1min").agg(aggregation_rules)

one_minute_bins["vwap_bid"] = one_minute_bins["weighted_bid"] / one_minute_bins["vol"]
one_minute_bins["vwap_ask"] = one_minute_bins["weighted_ask"] / one_minute_bins["vol"]
one_minute_bins = one_minute_bins[["vwap_bid", "vwap_ask"]]

one_minute_bins.rename(
    columns={"vwap_bid": "pre_VWAP_BID_7", "vwap_ask": "pre_VWAP_ASK_7"},
    inplace=True,
)

one_minute_bins_after_930 = one_minute_bins.between_time("09:30", "16:00")

# forward merger
post_merged_trades = pd.merge_asof(
    trades, Ask, on="time", direction="forward", suffixes=("", "_ask")
)
post_merged_trades = pd.merge_asof(
    post_merged_trades, Bid, on="time", direction="forward", suffixes=("", "_bid")
)

post_merged_trades["weighted_bid"] = (
    post_merged_trades["price_bid"] * post_merged_trades["vol"]
)
post_merged_trades["weighted_ask"] = (
    post_merged_trades["price_ask"] * post_merged_trades["vol"]
)
aggregation_rules = {
    "price": "mean",
    "vol": "sum",
    "weighted_bid": "sum",
    "weighted_ask": "sum",
    "price_bid": "mean",
    "price_ask": "mean",
}

one_minute_bins_post = post_merged_trades.resample("1min", on="time").agg(
    aggregation_rules
)
one_minute_bins_post["vwap_bid"] = (
    one_minute_bins_post["weighted_bid"] / one_minute_bins_post["vol"]
)
one_minute_bins_post["vwap_ask"] = (
    one_minute_bins_post["weighted_ask"] / one_minute_bins_post["vol"]
)
one_minute_bins_post = one_minute_bins_post[["vwap_bid", "vwap_ask"]]

one_minute_bins_post.rename(
    columns={"vwap_bid": "post_VWAP_BID_7", "vwap_ask": "post_VWAP_ASK_7"},
    inplace=True,
)

one_minute_bins_post_after_930 = one_minute_bins_post.between_time("09:30", "16:00")

# Merge the filtered DataFrames
dfs_to_merge = []
for name, df in aggregated_data.items():
    if not df.empty:
        dfs_to_merge.append(df)

if not one_minute_bins_after_930.empty:
    dfs_to_merge.append(one_minute_bins_after_930)

if not one_minute_bins_post_after_930.empty:
    dfs_to_merge.append(one_minute_bins_post_after_930)

# Merge the filtered DataFrames
if dfs_to_merge:
    merged_df = pd.concat(dfs_to_merge, axis=1)
    merged_df_reset = merged_df.reset_index()
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 10)
    
    for col in merged_df_reset.columns:
        if merged_df_reset[col].dtype == "object":
            merged_df_reset[col] = merged_df_reset[col].astype(str)
    
    datetime_columns = []
    for col in merged_df_reset.columns:
        if merged_df_reset[col].dtype == "datetime64[ns]":
            merged_df_reset[col] = merged_df_reset[col].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            datetime_columns.append(col)
    
    output_file_path = "/home/taq/taq_allstocks/datetime_columns.txt"
    
    try:
        with open(output_file_path, "w") as f:
            for column in datetime_columns:
                f.write(f"{column}\n")
        print("Datetime column names have been successfully written to the file.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    
    print(f"Saving data to HDF5 file: {hdf5_variable_path}")
    with pd.HDFStore(hdf5_variable_path, mode="a", complevel=9, complib="zlib") as store:
        hdf5_key = f"/{args.stock_name}/day{args.day}/time_bars"
        store.append(hdf5_key, merged_df_reset, format="table", data_columns=True, index=False)
        print(f"Data successfully saved to HDF5 key: {hdf5_key}")
else:
    print("No DataFrames to merge. Skipping HDF5 save step.")
    # Create a text file with a message indicating empty time bars
    empty_bars_file_path = "/home/taq/taq_allstocks/empty_time_bars.txt"
    message = f"{args.stock_name} has empty time bars for {args.day}/{args.month}/{args.year}."
    
    try:
        with open(empty_bars_file_path, "w") as f:
            f.write(message)
        print(f"Message written to {empty_bars_file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
