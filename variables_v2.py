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
aggr_sells_1min = Sells_trades.resample("1min", on="time").apply(custom_agg_function)

aggr_buys_1min_df = pd.DataFrame(aggr_buys_1min, columns=["last_buyer_price_4"])
aggr_sells_1min_df = pd.DataFrame(aggr_sells_1min, columns=["last_seller_price_4"])

aggr_buys_1min_after_930 = aggr_buys_1min_df.between_time("09:30", "16:00")
aggr_sells_1min_after_930 = aggr_sells_1min_df.between_time("09:30", "16:00")

print(aggr_buys_1min_after_930)
print(aggr_sells_1min_after_930)

# 5.VWAP of trades (and separately buys/sells) over interval
def custom_agg_function(data):
    weighted_prices = (data["price"] * data["vol"]).sum()
    total_volume = data["vol"].sum()

    if total_volume == 0:
        return 0.0
    else:
        return weighted_prices / total_volume

vwap_trades_1min = trades.resample("1min", on="time").apply(custom_agg_function)
vwap_buys_1min = Buys_trades.resample("1min", on="time").apply(custom_agg_function)
vwap_sells_1min = Sells_trades.resample("1min", on="time").apply(custom_agg_function)

vwap_trades_1min_df = pd.DataFrame(vwap_trades_1min, columns=["vwap_trades_5"])
vwap_buys_1min_df = pd.DataFrame(vwap_buys_1min, columns=["vwap_buys_5"])
vwap_sells_1min_df = pd.DataFrame(vwap_sells_1min, columns=["vwap_sells_5"])

vwap_trades_1min_after_930 = vwap_trades_1min_df.between_time("09:30", "16:00")
vwap_buys_1min_after_930 = vwap_buys_1min_df.between_time("09:30", "16:00")
vwap_sells_1min_after_930 = vwap_sells_1min_df.between_time("09:30", "16:00")

print(vwap_trades_1min_after_930)
print(vwap_buys_1min_after_930)
print(vwap_sells_1min_after_930)


# 6. Simple average of trade prices (and separately buys/sells) over interval
average_trades_1min = trades.resample("1min", on="time").agg({"price": "mean"})
average_buys_1min = Buys_trades.resample("1min", on="time").agg({"price": "mean"})
average_sells_1min = Sells_trades.resample("1min", on="time").agg({"price": "mean"})


average_trades_1min.rename(
    columns={ "price": "av_price_trades_6" }, inplace=True,
)
average_buys_1min.rename(
    columns={ "price": "av_price_buys_6" }, inplace=True,
)
average_sells_1min.rename(
    columns={ "price": "av_price_sells_6" }, inplace=True,
)


average_trades_1min_after_930 = average_trades_1min.between_time("09:30", "16:00")
average_buys_1min_after_930 = average_buys_1min.between_time("09:30", "16:00")
average_sells_1min_after_930 = average_sells_1min.between_time("09:30", "16:00")

print(vwap_trades_1min_after_930)
print(average_buys_1min_after_930)
print(average_sells_1min_after_930)


# 7. Volume weighted average pre/post-trade bid/ask prices (measured just before each trade and weighted by the size of each trade)
#backward merger
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
    columns={ "vwap_bid": "pre_vwap_bid_7", "vwap_ask": "pre_vwap_ask_7" }, inplace=True,
)

one_minute_bins_after_930 = one_minute_bins.between_time("09:30", "16:00")
print(one_minute_bins_after_930)

#forward merger
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
    columns={ "vwap_bid": "post_vwap_bid_7", "vwap_ask": "post_vwap_ask_7" }, inplace=True,
)

one_minute_bins_post_after_930 = one_minute_bins_post.between_time("09:30", "16:00")
print(one_minute_bins_post_after_930)

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
    
trades.set_index("time", inplace=True)
Ask.set_index("time", inplace=True)
Bid.set_index("time", inplace=True)
twap_trades = trades.resample("1min").apply(calculate_twap_and_volume)
twap_asks = Ask.resample("1min").apply(calculate_twap_and_volume)
twap_bids = Bid.resample("1min").apply(calculate_twap_and_volume)

twap_trades.rename(
    columns={ "TWAP": "TWAP_trades_8" , "TWAV": "TWAV_trades_8"}, inplace=True,
)
twap_asks.rename(
    columns={ "TWAP": "TWAP_asks_8" , "TWAV": "TWAV_asks_8"}, inplace=True,
)
twap_bids.rename(
    columns={ "TWAP": "TWAP_bids_8" , "TWAV": "TWAV_bids_8"}, inplace=True,
)

twap_trades_after_930 = twap_trades.between_time("09:30", "16:00")
twap_asks_after_930 = twap_asks.between_time("09:30", "16:00")
twap_bids_after_930 = twap_bids.between_time("09:30", "16:00")


print(twap_trades_after_930)
print(twap_asks_after_930)
print(twap_bids_after_930)

# 9. Total Volume traded over interval
if trades.index.name == "time":
    trades.reset_index(inplace=True)

if Ask.index.name == "time":
    Ask.reset_index(inplace=True)

if Bid.index.name == "time":
    Bid.reset_index(inplace=True)

trades_resampled = trades.resample("1min", on="time").agg({"vol": "sum"})
Buys_trades_resampled = Buys_trades.resample("1min", on="time").agg({"vol": "sum"})
Sells_trades_resampled = Sells_trades.resample("1min", on="time").agg({"vol": "sum"})
Ask_resampled = Ask.resample("1min", on="time").agg({"vol": "sum"})
Bid_resampled = Bid.resample("1min", on="time").agg({"vol": "sum"})

trades_resampled.rename(
    columns={ "vol": "tvol_trades_9"}, inplace=True,
)
trades_resampled_after_930 = trades_resampled.between_time("09:30", "16:00")
print(trades_resampled_after_930)

#10. Number of bid/ask price/volume changes over interval
def count_changes(series):
    price_changes = series.diff()
    num_changes = (price_changes != 0).sum()
    return num_changes


trades_changes = trades.resample("1min", on="time").apply(
    {"price": count_changes, "vol": count_changes}
)
Ask_changes = Ask.resample("1min", on="time").apply(
    {"price": count_changes, "vol": count_changes}
)
Bid_changes = Bid.resample("1min", on="time").apply(
    {"price": count_changes, "vol": count_changes}
)
trades_changes.rename(
    columns={ "price": "no_dp_trades_10", "vol" : "no_dv_trades_10"}, inplace=True,
)

trades_changes_after_930 = trades_changes.between_time("09:30", "16:00")
print(trades_changes_after_930)

#Merge first 20 variables into a single dataframe
merged_df = pd.concat([
    trades_1min_after_930, asks_1min_after_930, bids_1min_after_930,
    aggr_buys_1min_after_930, aggr_sells_1min_after_930,
    vwap_trades_1min_after_930, vwap_buys_1min_after_930, vwap_sells_1min_after_930,
    average_trades_1min_after_930, average_buys_1min_after_930, average_sells_1min_after_930,
    one_minute_bins_after_930, one_minute_bins_post_after_930,
    twap_trades_after_930, twap_asks_after_930, twap_bids_after_930,
    trades_resampled_after_930, trades_changes_after_930
], axis=1)

print(merged_df)
