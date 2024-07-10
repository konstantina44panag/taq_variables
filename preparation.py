import pandas as pd
import numpy as np
import tables
from sign_algorithms import TradeAnalyzer
import logging
import time
import sys
import traceback
from datetime import datetime
import polars as pl
from numba import njit

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
            if corr_pattern in col:
                data["corr"] = dataset.col(col)
            elif cond_pattern in col:
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

#def check_price_column_for_more_than_17_chars(df):
#    """Check if any value in the PRICE column has more than 16 characters."""
#    if "PRICE" in df.columns:
#        price_col = df["PRICE"].to_numpy()
#
#        lengths = np.vectorize(len)(price_col)
#        
#        if np.any(lengths > 17):
#            print("Values in the PRICE column with more than 16 characters:")
#            values_with_more_than_16_chars = price_col[lengths > 16]
#            for value in values_with_more_than_16_chars:
#                print(value)
#        else:
#            print("No value in the PRICE column has more than 16 characters.")
#    else:
#        print("PRICE column not found in the DataFrame.")


def handle_duplicates(df, key_col, value_cols, sum_col=None, other_cols=None):
    """
    Handle duplicates using Polars by taking the median for the value columns, the sum for the sum_col,
    and the last value for other columns. If other_cols is not provided, only the value_cols and sum_col 
    will be aggregated.

    """
    pl_df = pl.from_pandas(df)

    agg_exprs = [pl.col(col).median().alias(col) for col in value_cols]
    
    if sum_col:
        agg_exprs.extend([pl.col(col).sum().alias(col) for col in sum_col])
        
    if other_cols:
        agg_exprs.extend([pl.col(col).last().alias(col) for col in other_cols])
    
    result = pl_df.groupby(key_col).agg(agg_exprs)

    result_df = result.to_pandas().reset_index()
    result_df = result_df.sort_values(by=key_col).reset_index(drop=True)

    return result_df


def identify_retail(z):
            if 0 < z < 0.4 or 0.6 < z < 1:
                return 'retail trade'
            else:
                return 'non-retail trade'

   
def calculate_returns_shift(df, price_col='price', time_col='time', additional_cols=[]):
 
    prices = df[price_col].values
    times = df[time_col].values
    
    returns = np.zeros_like(prices)
    returns[1:] = np.log(prices[1:] / prices[:-1])

    returns_df = pd.DataFrame({
        'time': times,
        'returns': returns
    })

    if additional_cols:
        for col in additional_cols:
            returns_df[col] = df[col].values

    return returns_df

@njit
def rolling_median_exclude_self(series, window):
    medians = []
    for i in range(len(series)):
        if i < window // 2 or i >= len(series) - window // 2:
            medians.append(np.nan)
        else:
            window_data = np.delete(series[i - window // 2:i + window // 2 + 1], window // 2)
            medians.append(np.median(window_data))
    return np.array(medians)

@njit
def rolling_mad_exclude_self(series, window):
    mads = []
    for i in range(len(series)):
        if i < window // 2 or i >= len(series) - window // 2:
            mads.append(np.nan)
        else:
            window_data = np.delete(series[i - window // 2:i + window // 2 + 1], window // 2)
            median = np.median(window_data)
            mad = np.mean(np.abs(window_data - median))
            mads.append(mad)
    return np.array(mads)

dummy_data = np.random.rand(100)
_ = rolling_median_exclude_self(dummy_data, 5)
_ = rolling_mad_exclude_self(dummy_data, 5)

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

        decode_start_time = time.time()
        trades = decode_byte_strings(trades)
        nbbos = decode_byte_strings(nbbos)
        decode_end_time = time.time()

        logging.info("Cleaning data")
        #Formatting
        #trades
        format_start_time = time.time()
        trades["TIME_M"] = np.array(trades["TIME_M"], dtype=np.float64)
        trades = convert_float_to_datetime(trades, "TIME_M", base_date) 
        trades['PRICE'] = np.array(trades['PRICE'], dtype=np.float64)
        trades['SIZE'] = np.array(trades['SIZE'], dtype=np.float64)
        trades["EX"] = trades["EX"].astype(str)
        trades["corr"] = trades["corr"].astype(str)
        trades["cond"] = trades["cond"].astype(str)

        trades.rename(columns={
            "TIME_M": "time",
            "PRICE": "price",
            "SIZE": "vol"
        }, inplace=True)

        mask = (~np.isnan(trades['price'].values)) & (trades['price'].values != 0)
        trades = trades.loc[mask]
        
        #nbbo
        nbbos["TIME_M"] = np.array(nbbos["TIME_M"], dtype=np.float64)
        nbbos = convert_float_to_datetime(nbbos, "TIME_M", base_date)

        columns_to_convert = [
            "Best_AskSizeShares",
            "Best_BidSizeShares",
            "BEST_ASK",
            "BEST_BID",
        ]
        for col in columns_to_convert:
            nbbos[col] = np.array(nbbos[col], dtype=np.float64)
        nbbos["qu_cond"] = nbbos["qu_cond"].astype(str)
        nbbos.rename(columns={"TIME_M": "time"}, inplace=True)

        mask = (~np.isnan(nbbos['BEST_ASK'].values)) & (~np.isnan(nbbos['BEST_BID'].values)) & (nbbos['BEST_ASK'].values != 0) & (nbbos['BEST_BID'].values != 0)
        nbbos = nbbos.loc[mask]
        
        format_end_time = time.time()

        #Data cleaning
        clean_only_start_time = time.time()
       
        #Clean trades
        pl_trades = pl.from_pandas(trades) 
        x1 = time.time()  
        pl_trades = pl_trades.filter(pl_trades['corr'].is_in(['00', '01', '02']))
        end_x1 = time.time()  
        x2 = time.time()
        conditions_to_remove = "BGJKLOTWZ"
        pl_trades = pl_trades.filter(~pl_trades['cond'].str.contains(f"[{conditions_to_remove}]"))
        trades = pl_trades.to_pandas()
        end_x2 = time.time()

        print(trades)
        print(nbbos)
        x3 = time.time()
        trades['rolling_median'] = rolling_median_exclude_self(trades['price'].values, 51)
        trades['rolling_mad'] = rolling_mad_exclude_self(trades['price'].values, 51)
        trades['exclude'] = np.abs(trades['price'] - trades['rolling_median']) > 10 * trades['rolling_mad']
        trades = trades[~trades['exclude']]
        end_x3 = time.time()
        
        x4 = time.time()
        trades = handle_duplicates(trades, key_col='datetime', value_cols=['price'], sum_col=['vol'], other_cols=['time', "corr", "cond", "EX"])
        end_x4 = time.time()
        #Clean nbbo
        y1 = time.time()
        pl_nbbos = pl.from_pandas(nbbos)
        pl_nbbos = pl_nbbos.filter(pl_nbbos['BEST_ASK'] >= pl_nbbos['BEST_BID'])
        end_y1 = time.time()
        y2 = time.time()
        pl_nbbos = pl_nbbos.with_columns((pl_nbbos['BEST_ASK'] - pl_nbbos['BEST_BID']).alias('spread'))
        med_spread = pl_nbbos['spread'].median()
        pl_nbbos = pl_nbbos.filter(pl_nbbos['spread'] <= 50 * med_spread)
        end_y2 = time.time()

        y3 = time.time()
        pl_nbbos = pl_nbbos.with_columns(((pl_nbbos['BEST_BID'] + pl_nbbos['BEST_ASK']) / 2).alias('midpoint'))
        nbbos = pl_nbbos.to_pandas()
        end_y3 = time.time()
        y4 = time.time()
        nbbos['rolling_median'] = rolling_median_exclude_self(nbbos['midpoint'].values, 51)
        nbbos['rolling_mad'] = rolling_mad_exclude_self(nbbos['midpoint'].values, 51)
        nbbos['exclude'] = np.abs(nbbos['midpoint'] - nbbos['rolling_median']) > 10 * nbbos['rolling_mad']
        nbbos = nbbos[~nbbos['exclude']]
        end_y4 = time.time()
        y5 = time.time()
        nbbos = handle_duplicates(nbbos, key_col='datetime', value_cols=['BEST_ASK', 'BEST_BID', 'midpoint'],  sum_col=['Best_AskSizeShares', 'Best_BidSizeShares'], other_cols=['time', 'qu_cond'])
        end_y5 = time.time()
        clean_only_end_time = time.time()

        #Define the dataframes for variable calculations
        #Define the Ask and Bid
        Ask = nbbos[
            ["time", "datetime", "BEST_ASK", "Best_AskSizeShares", "qu_cond"]
        ].copy()
        Ask.rename(columns={"BEST_ASK": "price", "Best_AskSizeShares": "vol"}, inplace=True)
        Bid = nbbos[
            ["time", "datetime", "BEST_BID", "Best_BidSizeShares", "qu_cond"]
        ].copy()
        Bid.rename(columns={"BEST_BID": "price", "Best_BidSizeShares": "vol"}, inplace=True)
     
        #Define the Midpoint
        Midpoint = nbbos[['datetime', 'midpoint']].copy()
        Midpoint.rename(
            columns={"datetime" : "time", "midpoint": "price"}, inplace=True
        )

        #Define the nbbo_signs
        nbbo_start_time = time.time()
        best_bid = nbbos['BEST_BID'].values
        best_bid_size_shares = nbbos['Best_BidSizeShares'].values
        best_ask = nbbos['BEST_ASK'].values
        best_ask_size_shares = nbbos['Best_AskSizeShares'].values

        bid_change = np.diff(best_bid, prepend=best_bid[0]) != 0
        bid_size_change = np.diff(best_bid_size_shares, prepend=best_bid_size_shares[0]) != 0
        ask_change = np.diff(best_ask, prepend=best_ask[0]) != 0
        ask_size_change = np.diff(best_ask_size_shares, prepend=best_ask_size_shares[0]) != 0

        sign = (bid_change | bid_size_change).astype(int) - (ask_change | ask_size_change).astype(int)
        vol = np.where(sign == 1, best_bid_size_shares, np.where(sign == -1, best_ask_size_shares, 0))

        nbbos['sign'] = sign
        nbbos['vol_sign'] = vol
        nbbo_signs = nbbos[['datetime', 'sign', 'vol_sign']].copy()
        nbbo_signs.rename(
            columns={"datetime": "time", "sign": "returns", "vol_sign":"vol"}, inplace=True
        )
        nbbo_end_time = time.time() 

        #Create a value column
        trades['value'] = trades['price'] * trades['vol'] 
        Ask['value'] = Ask['price'] * Ask['vol'] 
        Bid['value'] = Bid['price'] * Bid['vol']

        #Trade Signs estimation

        trsigns_start_time = time.time()
        logging.info("Estimating trade signs")
        analyzer = TradeAnalyzer(trades, Ask, Bid)
        tradessigns = analyzer.classify_trades()
        trsigns_end_time = time.time()
        trsigns_time = trsigns_end_time - trsigns_start_time

        logging.info("More preparation")
        trades.sort_values(by="datetime", inplace=True)
        tradessigns.sort_values(by='datetime', inplace=True)
        Ask.sort_values(by="datetime", inplace=True)
        Bid.sort_values(by="datetime", inplace=True)
        Midpoint.sort_values(by="time", inplace=True) 

        trades.drop(columns=["time"], inplace=True)
        trades.rename(columns={"datetime": "time"}, inplace=True)
        tradessigns.drop(columns=["time", "time_org"], inplace=True)
        tradessigns.rename(columns={"datetime": "time"}, inplace=True)
        Ask.drop(columns=["time"], inplace=True)
        Ask.rename(columns={"datetime": "time"}, inplace=True)
        Bid.drop(columns=["time"], inplace=True)
        Bid.rename(columns={"datetime": "time"}, inplace=True)
        
        #Define the Buys_trades and Sell_trades
        buyes_sells_start_time = time.time()

        Buys_trades = tradessigns[tradessigns["Initiator"] == 1].copy()
        Sells_trades = tradessigns[tradessigns["Initiator"] == -1].copy()
        
        #Define the Retail_trades
        Retail_trades = trades.loc[trades["EX"] == "D"].copy()
        Retail_trades['Z'] = 100 * (Retail_trades['price'] % 0.01)        
        Retail_trades['trade_type'] = Retail_trades['Z'].apply(identify_retail)
        Retail_trades = Retail_trades[Retail_trades['trade_type'] == 'retail trade'].drop(columns=['trade_type'])

        #Define the Oddlot_trades
        target_date = datetime(2014, 1, 1)
        Oddlot_trades = trades[(trades['time'] >= target_date) & (trades['cond'] == "I")].copy()
        buyes_sells_end_time = time.time()
        #Define the Returns
        returns_start_time = time.time()

        trade_returns = calculate_returns_shift(trades, price_col='price', time_col='time', additional_cols=['vol'])
        midprice_returns = calculate_returns_shift(Midpoint, price_col='price', time_col='time')
        returns_end_time = time.time()

        #Define the trade_signs
        trade_signs = tradessigns[
            ["time", "Initiator", "vol"]].copy()
        trade_signs.rename(columns={"Initiator": "returns"}, inplace=True)
        sort_start_time = time.time()

        
        sort_end_time = time.time()

        with open("preparation_timeanalysis.txt", "a") as f:
            f.write(f"Stock: {stock_name}\n")
            f.write(f"Load time: {load_time} seconds\n")
            f.write(f"decode time: {decode_end_time - decode_start_time} seconds\n")
            f.write(f"format time: {format_end_time - format_start_time} seconds\n")
            f.write(f"Clean time: {clean_only_end_time - clean_only_start_time} seconds\n")
            f.write(f"x1 time: {end_x1 - x1} seconds\n")
            f.write(f"x2 time: {end_x2 - x2} seconds\n")
            f.write(f"x3 time: {end_x3 - x3} seconds\n")
            f.write(f"x4 time: {end_x4 - x4} seconds\n")
            f.write(f"y1 time: {end_y1 - y1} seconds\n")
            f.write(f"y2 time: {end_y2 - y2} seconds\n")
            f.write(f"y3 time: {end_y3 - y3} seconds\n")
            f.write(f"y4 time: {end_y4 - y4} seconds\n")
            f.write(f"y5 time: {end_y5 - y5} seconds\n")
            f.write(f"nbbo sign time: {nbbo_end_time - nbbo_start_time} seconds\n")
            f.write(f"buys_sells time: {buyes_sells_end_time - buyes_sells_start_time} seconds\n")
            f.write(f"returns time: {returns_end_time - returns_start_time} seconds\n")
            f.write(f"sort time: {sort_end_time - sort_start_time} seconds\n")
            f.write(f"TradeSigns time: {trsigns_time} seconds\n")

        return trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
