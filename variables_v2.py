#!/usr/bin/env python3.11
"""This script computes important variables but first passes arguments to preparation.py."""

import pandas as pd
import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser(
    description="Prepare datasets for trade sign analysis and variable estimation."
)
parser.add_argument("hdf5_file_path", type=str, help="The path to the HDF5 file.")
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

subprocess.run([
    "python3.11", "preparation.py",
    args.hdf5_file_path,
    args.ctm_dataset_path,
    args.complete_nbbo_dataset_path
])


from preparation import trades, Buys_trades, Sells_trades, Ask, Bid, tradeswithsign


# 1,2,3. The last trade/bid/ask price, volume, and timestamp
trades_1min = trades.resample("1min", on="time").agg(
    {"price": "last", "vol": "last", "time": "last"}
)

asks_1min = Ask.resample("1min", on="time").agg(
    {"price": "last", "vol": "last", "time": "last"}
)

bids_1min = Bid.resample("1min", on="time").agg(
    {"price": "last", "vol": "last", "time": "last"}
)

# Rename columns
trades_1min.rename(
    columns={
        "price": "last_trade_price",
        "vol": "last_trade_vol",
        "time": "last_trade_time",
    },
    inplace=True,
)
asks_1min.rename(
    columns={
        "price": "last_ask_price",
        "vol": "last_ask_vol",
        "time": "last_ask_time",
    },
    inplace=True,
)
bids_1min.rename(
    columns={
        "price": "last_bid_price",
        "vol": "last_bid_vol",
        "time": "last_bid_time",
    },
    inplace=True,
)

trades_1min_after_930 = trades_1min.between_time("09:30", "16:00")
asks_1min_after_930 = asks_1min.between_time("09:30", "16:00")
bids_1min_after_930 = bids_1min.between_time("09:30", "16:00")

print(trades_1min_after_930)
print(asks_1min_after_930)
print(bids_1min_after_930)



# 4.Last price at which an aggressive buy/sell order for $10,000 would execute
Buys_trades["value"] = Buys_trades["price"] * Buys_trades["vol"]
Sells_trades["value"] = Sells_trades["price"] * Sells_trades["vol"]


def custom_agg_function(data):
    filtered_trades = data[data["value"] >= 10000]
    if not filtered_trades.empty:
        return filtered_trades.iloc[-1]["price"]
    else:
        return np.nan

aggr_buys_1min = Buys_trades.resample("1min", on="time").apply(custom_agg_function)
aggr_buys_1min = aggr_buys_1min.rename("aggressive buyer's price")
agrr_sells_1min = Sells_trades.resample("1min", on="time").apply(custom_agg_function)
agrr_sells_1min = agrr_sells_1min.rename("aggressive seller's price")


aggr_buys_1min_after_930 = aggr_buys_1min.between_time("09:30", "16:00")
agrr_sells_1min_after_930= agrr_sells_1min.between_time("09:30", "16:00")

print(aggr_buys_1min_after_930)
print(agrr_sells_1min _after_930)

# 5.VWAP of trades (and separately buys/sells) over interval
def custom_agg_function(data):
    weighted_prices = (data["price"] * data["vol"]).sum()
    total_volume = data["vol"].sum()

    if total_volume == 0:
        return 0.0
    else:
        return weighted_prices / total_volume


vwap_trades_1min = trades.resample("1min", on="time").apply(custom_agg_function)
vwap_trades_1min = vwap_trades_1min.rename("volume_weighted_price")

vwap_buys_1min = Buys_trades.resample("1min", on="time").agg(custom_agg_function)
vwap_buys_1min = vwap_buys_1min.rename("volume_weighted_price")

vwap_sells_1min = Sells_trades.resample("1min", on="time").agg(custom_agg_function)
vwap_sells_1min = vwap_sells_1min.rename("volume_weighted_price")

vwap_trades_1min_after_930 = vwap_trades_1min.between_time("09:30", "16:00")
vwap_buys_1min_after_930 = vwap_buys_1min.between_time("09:30", "16:00")
vwap_sells_1min_after_930 = vwap_sells_1min.between_time("09:30", "16:00")

print(vwap_trades_1min_after_930)
print(vwap_buys_1min_after_930)
print(vwap_sells_1min_after_930)


# 6. Simple average of trade prices (and separately buys/sells) over interval
average_trades_1min = trades.resample("1min", on="time").agg({"price": "mean"})
average_buys_1min = Buys_trades.resample("1min", on="time").agg({"price": "mean"})
average_sells_1min = Sells_trades.resample("1min", on="time").agg({"price": "mean"})

average_trades_1min_after_930 = average_trades_1min.between_time("09:30", "16:00")
average_buys_1min_after_930 = average_buys_1min.between_time("09:30", "16:00")
average_sells_1min_after_930 = average_sells_1min.between_time("09:30", "16:00")

print(vwap_trades_1min_after_930)
print(average_buys_1min_after_930)
print(average_sells_1min_after_930)

# 8. Time weighted version of bid/ask prices and size (in dollars) of best bid and ask
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

twap_trades = trades.resample("1min").apply(calculate_twap_and_volume)
twap_asks = Ask.resample("1min").apply(calculate_twap_and_volume)
twap_bids = Bid.resample("1min").apply(calculate_twap_and_volume)

twap_trades_after_930 = twap_trades.between_time("09:30", "16:00")
twap_asks_after_930 = twap_asks.between_time("09:30", "16:00")
twap_bids_after_930 = twap_bids.between_time("09:30", "16:00")

print(twap_trades_after_930)
print(twap_asks_after_930)
print(twap_bids_after_930)
