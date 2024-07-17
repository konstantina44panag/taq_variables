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
pd.set_option('display.max_rows', 100)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
class NoTradesException(Exception):
    pass
class NoNbbosException(Exception):
    pass


# Function definitions
def print_debug_info(df, name):
    print(f"\n{name}")
    print(df.head())
    print(df.columns)
    print(f"DataFrame shape: {df.shape}")

#def convert_float_to_datetime: Conversion of float timestamps to datetime object timestamps
def convert_float_to_datetime(df, float_column, base_date):
    """Converts float time values to datetime objects based on a base date."""
    midnight = pd.to_datetime(base_date + " 00:00:00")
    df["timedelta"] = pd.to_timedelta(df[float_column], unit="s")
    df["datetime"] = midnight + df["timedelta"]
    df.drop(columns=["timedelta"], inplace=True)
    return df
    
#def load_dataset: For loading the trades dataframe from the ctm files, the necessary columns for cleaning and variables are TIME_M, PRICE, SIZE, TR_CORR, SALE_COND, in some files the names change to TRCORR, TRCOND, 
#so I identify the columns by the patterns CORR, COND
#The data are loaded to the trades dataframe
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
        
#def load_dataset_with_exclusion: For loading the nbbo dataframe from the complete_nbbo files, the necessary columns for cleaning and variables are TIME_M, BEST_ASK, BEST_BID, BestAskShares, BestBidShares, QU_COND in some files the names change to QUOTE_COND 
#so I identify the column by the pattern COND and excluding the pattern NBBO, since NBBO_COND is a different column than QU_COND
#The data are loaded to the nbbos dataframe
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
        
#def decode_byte_strings: The data in HDF5 files have the type byte strings, I decode the data to string type    
def decode_byte_strings(df):
    """Decode byte strings in all columns of the dataframe."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    return df


#def handle_duplicates: For cleaning data with the same timestamps, by using the median price and aggregating the volume
def handle_duplicates(pl_df, key_col, value_cols, sum_col=None, other_cols=None, join_col=None):
    """
    Handle duplicates using Polars by taking the median for the value columns, the sum for the sum_col,
    and the last value for other columns. If other_cols is not provided, only the value_cols and sum_col 
    will be aggregated.

    """
    if pl_df.shape[0] < 2:
        return pl_df

    duplicates_mask = pl_df.with_columns(pl.col(key_col).is_duplicated().alias("is_duplicated"))
    duplicates = duplicates_mask.filter(pl.col("is_duplicated")).drop("is_duplicated")
    non_duplicates = duplicates_mask.filter(~pl.col("is_duplicated")).drop("is_duplicated")

    agg_exprs = [pl.col(col).median().alias(col) for col in value_cols]
    
    if sum_col:
        agg_exprs.extend([pl.col(col).sum().alias(col) for col in sum_col])
        
    if other_cols:
        agg_exprs.extend([pl.col(col).last().alias(col) for col in other_cols])
    if join_col:
        agg_exprs.extend([pl.col(col).str.concat(",") for col in join_col])
   
    aggregated_duplicates = duplicates.group_by(key_col).agg(agg_exprs).sort(key_col)
    aggregated_duplicates = aggregated_duplicates.select(non_duplicates.columns)
    result_df = pl.concat([non_duplicates, aggregated_duplicates]).sort(key_col)
    return result_df

#def identify_retail: For finding the retail trades from the trades dataframe
def identify_retail(z):
            if 0 < z < 0.4 or 0.6 < z < 1:
                return 'retail trade'
            else:
                return 'non-retail trade'

#def calculate_returns_shift: For calculating returns 
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

#For computing the rolling median for the price value of a dataframe, in order to apply the cleaning step A2 in TAQ Cleaning Techniques
@njit
def rolling_median_exclude_self(a, W):
    half_window = W // 2
    medians = np.full(a.size, np.nan)
    
    for i in range(half_window, len(a) - half_window):
        if i < half_window or i >= len(a) - half_window:
            continue
        window_data = np.concatenate((a[i - half_window:i], a[i + 1:i + half_window + 1]))
        medians[i] = np.median(window_data)

    return medians

#For computing the rolling mad for the price value of a dataframe, in order to apply the cleaning step A2 in TAQ Cleaning Techniques
@njit
def rolling_mad_exclude_self(a, W):
    half_window = W // 2
    mads = np.full(a.size, np.nan)
    
    for i in range(half_window, len(a) - half_window):
        if i < half_window or i >= len(a) - half_window:
            continue
        window_data = np.concatenate((a[i - half_window:i], a[i + 1:i + half_window + 1]))
        median = np.median(window_data)
        mad = np.mean(np.abs(window_data - median))
        mads[i] = mad
    
    return mads

dummy_data = np.random.rand(100)
_ = rolling_median_exclude_self(dummy_data, 5)
_ = rolling_mad_exclude_self(dummy_data, 5)

#def prepare_datasets: Contains the loading of data to dataframes and applies the appropriate operations, this function is called by the python script variables_v4.py which then calculates variables
def prepare_datasets(hdf5_file_path, base_date, stock_name, year, month, day, ctm_dataset_path, complete_nbbo_dataset_path, hdf5_variable_path, prep_analysis_path=None, emp_analysis_path=None, var_analysis_path=None, prof_analysis_path=None):
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

        #Applying decoding
        decode_start_time = time.time()
        trades = decode_byte_strings(trades)
        nbbos = decode_byte_strings(nbbos)
        decode_end_time = time.time()

        logging.info("Cleaning data")
        
        #Applying formatting to trades
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

        #Applying formatting to nbbos
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
        
        format_end_time = time.time()

        #Data cleaning for trades
        clean_only_start_time = time.time()
       
        #Remove nan or zero prices, P2 cleaning step in Taq Cleaning Techniques
        mask = (~np.isnan(trades['price'].values)) & (trades['price'].values != 0)
        trades = trades.loc[mask]

        #switch to polars dataframe
        pl_trades = pl.from_pandas(trades)

        #Check for empty dataframe after the cleaning step, otherwise an error will occur on next operations. If the trades are empty the program returns the message and calculations are skipped for this stock-day
        if pl_trades.height == 0:
            print(f"No trades after cleaning techniques for {stock_name}")
            raise NoTradesException()

        #Cleaning step T1 
        pl_trades = pl_trades.filter(pl_trades['corr'].is_in(['00', '01', '02']))

        #Check for empty dataframe after the cleaning step
        if pl_trades.height == 0:
            print(f"No trades after cleaning techniques for {stock_name}")
            raise NoTradesException()
            
        #Cleaning step T2
        pl_trades = pl_trades.filter(~pl_trades['cond'].str.contains('B|G|J|K|L|W|Z'))

        #Check for empty dataframe after the cleaning step
        if pl_trades.height == 0:
            print(f"No trades after cleaning techniques for {stock_name}")
            raise NoTradesException()
        
        #Cleaning step T3
        pl_trades = handle_duplicates(pl_trades, key_col=['datetime'], value_cols=['price'], sum_col=['vol'], other_cols=['time', "corr"], join_col=['cond', "EX"])

        #switch to pandas
        trades = pl_trades.to_pandas()
        trades.reset_index(drop=True)

        #Cleaning step A2, substitutes T4
        trades['rolling_median'] = rolling_median_exclude_self(trades['price'].values, 50)
        trades['rolling_mad'] = rolling_mad_exclude_self(trades['price'].values, 50)
        trades['exclude'] = np.abs(trades['price'] - trades['rolling_median']) > 10 * trades['rolling_mad']
        trades = trades[~trades['exclude']]
        trades = trades.drop(columns=['rolling_median', 'rolling_mad', 'exclude'])

        #Check for empty dataframe after the cleaning step
        if trades.empty:
            print(f"No trades after cleaning techniques for {stock_name}")
            raise NoTradesException()

        
        #Cleaning nbbos
        #Remove nan or zero quotes, cleaning step P2
        mask = (~np.isnan(nbbos['BEST_ASK'].values)) & (~np.isnan(nbbos['BEST_BID'].values)) & (nbbos['BEST_ASK'].values != 0) & (nbbos['BEST_BID'].values != 0)
        nbbos = nbbos.loc[mask]        
        #switch to polars
        pl_nbbos = pl.from_pandas(nbbos)
        
        #Check for empty dataframe after the cleaning step
        if pl_nbbos.height == 0:
            print(f"No nbbos after cleaning techniques for {stock_name}")
            raise NoNbbosException()
        
        #Cleaning Step Q1
        pl_nbbos = handle_duplicates(pl_nbbos, key_col='datetime', value_cols=['BEST_ASK', 'BEST_BID'],  sum_col=['Best_AskSizeShares', 'Best_BidSizeShares'], other_cols=['time', 'qu_cond'])

        #Cleaning Step Q2
        pl_nbbos = pl_nbbos.filter(pl_nbbos['BEST_ASK'] >= pl_nbbos['BEST_BID'])
        
        if pl_nbbos.height == 0:
            print(f"No nbbos after cleaning techniques for {stock_name}")
            raise NoNbbosException()

        #Cleaning Step Q3
        pl_nbbos = pl_nbbos.with_columns((pl_nbbos['BEST_ASK'] - pl_nbbos['BEST_BID']).alias('spread'))
        med_spread = pl_nbbos['spread'].median()
        pl_nbbos = pl_nbbos.filter(pl_nbbos['spread'] <= 50 * med_spread)

        #Check for empty dataframe after the cleaning step
        if pl_nbbos.height == 0:
            print(f"No nbbos after cleaning techniques for {stock_name}")
            raise NoNbbosException()
            
        #Create Midpoint
        pl_nbbos = pl_nbbos.with_columns(((pl_nbbos['BEST_BID'] + pl_nbbos['BEST_ASK']) / 2).alias('midpoint'))
        #switch to pandas
        nbbos = pl_nbbos.to_pandas()

        #switch to pandas
        nbbos = pl_nbbos.to_pandas()
        nbbos.reset_index(drop=True)

        #Cleaning Step Q4
        nbbos['rolling_median'] = rolling_median_exclude_self(nbbos['midpoint'].values, 51)
        nbbos['rolling_mad'] = rolling_mad_exclude_self(nbbos['midpoint'].values, 51)
        nbbos['exclude'] = np.abs(nbbos['midpoint'] - nbbos['rolling_median']) > 10 * nbbos['rolling_mad']
        nbbos = nbbos[~nbbos['exclude']]
        nbbos = nbbos.drop(columns=['rolling_median', 'rolling_mad', 'exclude'])
        
        #Check for empty dataframe after the cleaning step
        if nbbos.empty:
            print(f"No nbbos after cleaning techniques for {stock_name}")
            raise NoNbbosException()

        clean_only_end_time = time.time()

        
        #Define the appropriate dataframes for variable calculation
        trades.drop
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
        trades.reset_index(drop=True, inplace=True)
        Ask.reset_index(drop=True, inplace=True)
        Bid.reset_index(drop=True, inplace=True)

        trsigns_start_time = time.time()
        logging.info("Estimating trade signs")
        analyzer = TradeAnalyzer(trades, Ask, Bid)
        tradessigns = analyzer.classify_trades()
        trsigns_end_time = time.time()
        trsigns_time = trsigns_end_time - trsigns_start_time

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

        
        #Define trade specific dataframes
        specific_df_start_time = time.time()
        
        #Define the Buys_trades and Sell_trades dataframes
        Buys_trades = tradessigns[tradessigns["Initiator"] == 1].copy()
        Sells_trades = tradessigns[tradessigns["Initiator"] == -1].copy()
        
        #Define the Retail_trades dataframe
        Retail_trades = trades.loc[trades["EX"] == "D"].copy()
        Retail_trades['Z'] = 100 * (Retail_trades['price'] % 0.01)        
        Retail_trades['trade_type'] = Retail_trades['Z'].apply(identify_retail)
        Retail_trades = Retail_trades[Retail_trades['trade_type'] == 'retail trade'].drop(columns=['trade_type'])

        #Define the Oddlot_trades dataframe
        target_date = datetime(2014, 1, 1)
        Oddlot_trades = trades[(trades['time'] >= target_date) & (trades['cond'] == "I")].copy()
        specific_df_end_time = time.time()
        
        #Define the Returns dataframe
        returns_start_time = time.time()
        trade_returns = calculate_returns_shift(trades, price_col='price', time_col='time', additional_cols=['vol'])
        midprice_returns = calculate_returns_shift(Midpoint, price_col='price', time_col='time')
        returns_end_time = time.time()

        #Define the trade_signs dataframe
        trade_signs = tradessigns[
            ["time", "Initiator", "vol"]].copy()
        trade_signs.rename(columns={"Initiator": "returns"}, inplace=True)
    
        #Write the time analysis
        if prep_analysis_path is not None:
            with open(prep_analysis_path, "a") as f:
                f.write(f"Stock: {stock_name}\n")
                f.write(f"Day: {base_date}\n")
                f.write(f"Load time: {load_time} seconds\n")
                f.write(f"decode time: {decode_end_time - decode_start_time} seconds\n")
                f.write(f"format time: {format_end_time - format_start_time} seconds\n")
                f.write(f"Clean time: {clean_only_end_time - clean_only_start_time} seconds\n")
                f.write(f"nbbo sign time: {nbbo_end_time - nbbo_start_time} seconds\n")
                f.write(f"specific trades time: {specific_df_end_time - specific_df_start_time} seconds\n")
                f.write(f"returns time: {returns_end_time - returns_start_time} seconds\n")
                f.write(f"TradeSigns time: {trsigns_time} seconds\n")

        return trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs

    except  NoTradesException:
        return None
    except  NoNbbosException:
        return None

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

