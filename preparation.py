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
class NoTradesException(Exception):
    pass
class NoNbbosException(Exception):
    pass


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
def load_trades_dataset(
    hdf_file,
    dataset_path,
    columns_of_interest,
    corr_pattern="CORR",
    suf_pattern="SUF",
    cond_pattern="COND",
):
    """Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists."""
    try:
        table_path=f"{dataset_path}table"
        dataset = hdf_file.get_node(table_path)

        column_names =  hdf_file.get_node(dataset_path)._v_attrs["column_names"]
        data = {}
        for col in columns_of_interest:
            if col in column_names:
                data[col] = dataset.col(col)

        for col in column_names:
            if corr_pattern in col:
                data["corr"] = dataset.col(col)
            elif suf_pattern in col:
                data["suffix"] = dataset.col(col)
            elif cond_pattern in col:
                data["cond"] = dataset.col(col)

        df = pd.DataFrame(data)
        return df

    except tables.NoSuchNodeError as e:
        return None
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
        
#def load_dataset_with_exclusion: For loading the nbbo dataframe from the complete_nbbo files, the necessary columns for cleaning and variables are TIME_M, BEST_ASK, BEST_BID, BestAskShares, BestBidShares, QU_COND in some files the names change to QUOTE_COND 
#so I identify the column by the pattern COND and excluding the pattern NBBO, since NBBO_COND is a different column than QU_COND
#The data are loaded to the nbbos dataframe
def load_quotes_dataset(
    hdf_file,
    dataset_path,
    columns_of_interest,
    cond_pattern="COND",
    suf_pattern="SUF",
    exclude_pattern="NBBO",
):
    """Load specific dataset from HDF5 file using PyTables, ensuring necessary metadata exists."""
    try:
        table_path=f"{dataset_path}table"
        dataset = hdf_file.get_node(table_path)

        column_names=hdf_file.get_node(dataset_path)._v_attrs["column_names"]

        data = {}
        for col in columns_of_interest:
            if col in column_names:
                data[col] = dataset.col(col)

        for col in column_names:
            if suf_pattern in col:
                data["suffix"] = dataset.col(col)

            if cond_pattern in col and exclude_pattern not in col:
                data["qu_cond"] = dataset.col(col)

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


#def handle_duplicates: For cleaning data with the same timestamps, by using the median price and aggregating the volume
def handle_duplicates(pl_df, key_col, value_col=None, sum_col=None, other_col=None, join_col=None):
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
    agg_exprs = []
    if sum_col:
        agg_exprs.extend([pl.col(col).median().alias(col) for col in value_col])
    if sum_col:
        agg_exprs.extend([pl.col(col).sum().alias(col) for col in sum_col])   
    if other_col:
        agg_exprs.extend([pl.col(col).last().alias(col) for col in other_col])
    if join_col:
        agg_exprs.extend([pl.col(col).str.concat(",") for col in join_col])
   
    aggregated_duplicates = duplicates.group_by(key_col).agg(agg_exprs).sort(key_col)
    aggregated_duplicates = aggregated_duplicates.select(non_duplicates.columns)
    result_df = pl.concat([non_duplicates, aggregated_duplicates]).sort(key_col)
    return result_df

#def identify_retail: For finding the retail trades from the trades dataframe

def identify_retail(z):
    epsilon = 1e-12
    if epsilon < z < 1 - epsilon :
        return 'retail trade'
    else:
        return 'non-retail trade'
    
def identify_retail_old(z):
    epsilon = 1e-12
    if epsilon < z < 1 - epsilon :
        if z < 0.4 or z > 0.6 :
            return 'retail trade'
        else:
            return 'non-retail trade' 
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
    medians = np.full(a.size, np.nan)
    
    for i in range(a.size):
        start = max(0, i - W // 2)
        end = min(a.size, i + W // 2 + 1)
        if i == 0:
            window_data = a[i + 1:end]
        elif i == 1:
            window_data = np.concatenate((a[:i-1], a[i + 1:end]))
        elif i > 1 and i < a.size:
            window_data = np.concatenate((a[start:i-1], a[i + 1:end ]))
        elif i == a.size:
            window_data = a[start:i-1]
        if window_data.size > 0:
            medians[i] = np.median(window_data)

    return medians

#For computing the rolling mad for the price value of a dataframe, in order to apply the cleaning step A2 in TAQ Cleaning Techniques
@njit
def rolling_mad_exclude_self(a, W):
    mads = np.full(a.size, np.nan)
    
    for i in range(a.size):
        start = max(0, i - W // 2)
        end = min(a.size, i + W // 2 )
        if i == 0:
            window_data = a[i + 1:end]
        elif i == 1:
            window_data = np.concatenate((a[:i-1], a[i + 1:end]))
        elif i > 1 and i < a.size:
            window_data = np.concatenate((a[start:i-1], a[i + 1:end ]))
        elif i == a.size:
            window_data = a[start:i-1]
        if window_data.size > 0:
            median = np.median(window_data)
            mad = np.mean(np.abs(window_data - median))
            mads[i] = mad
    
    return mads

dummy_data = np.random.rand(50)
_ = rolling_median_exclude_self(dummy_data, 5)
_ = rolling_mad_exclude_self(dummy_data, 5)

#def prepare_datasets: Contains the loading of data to dataframes and applies the appropriate operations, this function is called by the python script variables_v4.py which then calculates variables
def prepare_datasets(hdf5_file_path, base_date, stock_name, s, year, month, day, ctm_dataset_path, complete_nbbo_dataset_path, hdf5_variable_path, prep_analysis_path=None, emp_analysis_path=None, var_analysis_path=None, prof_analysis_path=None):
    try:
        load_start_time = time.time()

        # Load datasets
        with tables.open_file(hdf5_file_path, "r") as hdf:
            trades = load_trades_dataset(
                hdf,
                ctm_dataset_path,
                ["TIME_M", "EX", "PRICE", "SIZE"],
                corr_pattern="CORR",
                suf_pattern="SUF",
                cond_pattern="COND",
            )
            if trades is None:
                raise NoTradesException()
            
            nbbos = load_quotes_dataset(
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
                suf_pattern="SUF",
                exclude_pattern="NBBO",
            )
            if nbbos is None:
                raise NoNbbosException()

        load_end_time = time.time()
        load_time = load_end_time - load_start_time

        #Applying decoding
        decode_start_time = time.time()
        trades = decode_byte_strings(trades)
        nbbos = decode_byte_strings(nbbos)
        decode_end_time = time.time()
        trades["suffix"] =  trades["suffix"].astype(str)
        nbbos["suffix"] = nbbos["suffix"].astype(str)
        trades = trades[trades["suffix"] == s]
        nbbos = nbbos[nbbos["suffix"] == s]
        trades.drop(columns=["suffix"], inplace=True)
        nbbos.drop(columns=["suffix"], inplace=True)
        if trades.empty:
            raise NoTradesException()
        if nbbos.empty:
            raise NoNbbosException()

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
        clean_start_time = time.time()
        #P2 cleaning step in Taq Cleaning Techniques google doc 
        mask = (~np.isnan(trades['price'].values)) & (trades['price'].values > 0)
        trades = trades[mask].copy()
        pl_trades = pl.from_pandas(trades)
        if pl_trades.height == 0:
            raise NoTradesException()

        #Cleaning step T1 
        pl_trades = pl_trades.filter(pl_trades['corr'].is_in(['00', '01', '02']))
        if pl_trades.height == 0:
            raise NoTradesException()
            
        #Cleaning step T3
        pl_trades = handle_duplicates(pl_trades, key_col=['datetime'], value_col=['price'], sum_col=['vol'], other_col=['time', "corr"], join_col=['cond', "EX"])
        trades = pl_trades.to_pandas()
        trades.reset_index(drop=True)
      
        #Cleaning nbbos according to Holden and Jacobsen(2014)
        mask = (
            ((nbbos['BEST_ASK'] <= 0) & (nbbos['BEST_BID'] <= 0)) |
            ((nbbos['Best_AskSizeShares'] <= 0) & (nbbos['Best_BidSizeShares'] <= 0)) |
            (np.isnan(nbbos['BEST_ASK'].values) & (np.isnan(nbbos['BEST_BID'].values))) |
            (np.isnan(nbbos['Best_AskSizeShares'].values) & (np.isnan(nbbos['Best_BidSizeShares'].values)))
        )
        nbbos = nbbos[~mask].copy()
        if nbbos.empty:
            raise NoNbbosException()

        best_bid = nbbos['BEST_BID'].to_numpy()
        best_bid_size = nbbos['Best_BidSizeShares'].to_numpy()
        best_ask = nbbos['BEST_ASK'].to_numpy()
        best_ask_size = nbbos['Best_AskSizeShares'].to_numpy()

        @njit
        def process_data(best_bid, best_bid_size, best_ask, best_ask_size):
            n = len(best_bid)
            midpoint = (best_bid + best_ask) / 2
            spread = best_ask - best_bid           

            for i in range(n):
                if best_ask[i] <= 0:
                    best_ask[i] = np.nan
                    best_ask_size[i] = np.nan
                if np.isnan(best_ask[i]):
                    best_ask_size[i] = np.nan
                if best_ask_size[i] <= 0:
                    best_ask[i] = np.nan
                    best_ask_size[i] = np.nan
                if np.isnan(best_ask_size[i]):
                    best_ask[i] = np.nan

                if best_bid[i] <= 0:
                    best_bid[i] = np.nan
                    best_bid_size[i] = np.nan
                if np.isnan(best_bid[i]):
                    best_bid_size[i] = np.nan
                if best_bid_size[i] <= 0:
                    best_bid[i] = np.nan
                    best_bid_size[i] = np.nan
                if np.isnan(best_bid_size[i]):
                    best_bid[i] = np.nan
           
            # Handle lmid, lm25, and lp25
            lmid = np.empty(n)
            lmid[0] = np.nan
            lmid[1:] = midpoint[:-1]
            lm25 = lmid - 2.5
            lp25 = lmid + 2.5
            
            # Apply spread conditions
            for i in range(n):
                if spread[i] > 5:
                    if best_bid[i] < lm25[i]:
                        best_bid[i] = np.nan
                        best_bid_size[i] = np.nan
                    if best_ask[i] > lp25[i]:
                        best_ask[i] = np.nan
                        best_ask_size[i] = np.nan
            
            return best_bid, best_bid_size, best_ask, best_ask_size, midpoint, spread

        # Process data
        best_bid, best_bid_size, best_ask, best_ask_size, midpoint, spread = process_data(best_bid, best_bid_size, best_ask, best_ask_size)
        # Update DataFrame
        nbbos['BEST_BID'], nbbos['Best_BidSizeShares'], nbbos['BEST_ASK'], nbbos['Best_AskSizeShares'], nbbos['midpoint'], nbbos['spread'] = best_bid, best_bid_size, best_ask, best_ask_size, midpoint, spread

        # Remove rows based on changes in specific columns
        mask = (
            (nbbos['BEST_ASK'] != nbbos['BEST_ASK'].shift(1)) |
            (nbbos['BEST_BID'] != nbbos['BEST_BID'].shift(1)) |
            (nbbos['Best_AskSizeShares'] != nbbos['Best_AskSizeShares'].shift(1)) |
            (nbbos['Best_BidSizeShares'] != nbbos['Best_BidSizeShares'].shift(1))
        )
        nbbos = nbbos[mask].copy()
        if nbbos.empty:
            raise NoNbbosException()
        
        #Cleaning Step Q1 &Also in H&J CODE in that order
        pl_nbbos = pl.from_pandas(nbbos)
        pl_nbbos = handle_duplicates(pl_nbbos, key_col='datetime', value_col=None,  sum_col=None, other_col=['time', 'BEST_ASK', 'BEST_BID', 'Best_AskSizeShares', 'Best_BidSizeShares', 'midpoint', 'spread'], join_col=['qu_cond'])
        nbbos = pl_nbbos.to_pandas()
        nbbos.reset_index(drop=True)
        
        #Define the appropriate dataframes for variable calculation
        #Define the Ask and Bid
        Ask = nbbos[
            ["time", "datetime", "BEST_ASK", "Best_AskSizeShares", "qu_cond"]
        ].copy()
        Ask.rename(columns={"BEST_ASK": "price", "Best_AskSizeShares": "vol"}, inplace=True)
        Bid = nbbos[
            ["time", "datetime", "BEST_BID", "Best_BidSizeShares", "qu_cond"]
        ].copy()
        Bid.rename(columns={"BEST_BID": "price", "Best_BidSizeShares": "vol"}, inplace=True)
        #Trade Signs estimation
        trades.reset_index(drop=True, inplace=True)
        Ask.reset_index(drop=True, inplace=True)
        Bid.reset_index(drop=True, inplace=True)

        trsigns_start_time = time.time()
        analyzer = TradeAnalyzer(trades, Ask, Bid)
        tradessigns = analyzer.classify_trades()
        trsigns_end_time = time.time()
        trsigns_time = trsigns_end_time - trsigns_start_time

        #sort dataframes and rename columns after trade signing
        tradessigns.sort_values(by='datetime', inplace=True)
        tradessigns.rename(columns={"time": "time_float"}, inplace=True)
        tradessigns.rename(columns={"datetime": "time"}, inplace=True)

        #Now that trades are matched with quote pairs, continue with data cleaning. 
        #Cleaning step, result of Q2 on trades:
        mask = tradessigns['ask'] > tradessigns['bid']
        tradessigns = tradessigns[mask].copy()
        #Cleaning Step Q2:
        pl_nbbos = pl_nbbos.filter(pl_nbbos['BEST_ASK'] > pl_nbbos['BEST_BID'])
        if pl_nbbos.height == 0:
            raise NoNbbosException()
        #The following data cleaning is done by Barndorff-Nielsen et al. (2009), Holden and Jacobsen apply the above methods in the same order
        #Cleaning step T4:
        tradessigns['spread'] = tradessigns['ask'] - tradessigns['bid']
        tradessigns['upper_bound'] = tradessigns['ask'] + tradessigns['spread']
        tradessigns['lower_bound'] = tradessigns['bid'] - tradessigns['spread']
        mask= (tradessigns['price'] <= tradessigns['upper_bound']) & (tradessigns['price'] >= tradessigns['lower_bound'])
        tradessigns = tradessigns[mask].copy()
        tradessigns.drop(columns=['upper_bound', 'lower_bound'], inplace=True)
        if tradessigns.empty:
            raise NoTradesException()
        tradessigns.sort_values(by='time', inplace=True)
        #Cleaning Step Q3
        med_spread = pl_nbbos['spread'].median()
        pl_nbbos = pl_nbbos.filter(pl_nbbos['spread'] <= 50 * med_spread)
        if pl_nbbos.height == 0:
            raise NoNbbosException()
        nbbos = pl_nbbos.to_pandas()
        #Cleaning Step Q4
        nbbos['rolling_median'] = rolling_median_exclude_self(nbbos['midpoint'].values, 50)
        nbbos['rolling_mad'] = rolling_mad_exclude_self(nbbos['midpoint'].values, 50)
        nbbos['exclude'] = np.abs(nbbos['midpoint'] - nbbos['rolling_median']) > 10 * nbbos['rolling_mad']
        nbbos = nbbos[~nbbos['exclude']]
        nbbos = nbbos.drop(columns=['rolling_median', 'rolling_mad', 'exclude'])
        if nbbos.empty:
            raise NoNbbosException()
        nbbos.sort_values(by='time', inplace=True)
        clean_end_time = time.time()

        #Define the trades, Ask and Bid, the Midpoint and nbbo signs after the cleaning
        trades = tradessigns
        Ask = nbbos[
            ["datetime", "BEST_ASK", "Best_AskSizeShares", "qu_cond"]
        ].copy()
        Ask.rename(columns={"BEST_ASK": "price", "Best_AskSizeShares": "vol", "datetime":"time"}, inplace=True)
        Bid = nbbos[
            ["datetime", "BEST_BID", "Best_BidSizeShares", "qu_cond"]
        ].copy()
        Bid.rename(columns={"BEST_BID": "price", "Best_BidSizeShares": "vol", "datetime":"time"}, inplace=True)
        Midpoint = nbbos[['datetime', 'midpoint']].copy()
        Midpoint.rename(
            columns={"datetime" : "time", "midpoint": "price"}, inplace=True
        )
        #Define the nbbo_signs
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

        #Create a value column
        trades['value'] = trades['price'] * trades['vol'] 
        Ask['value'] = Ask['price'] * Ask['vol'] 
        Bid['value'] = Bid['price'] * Bid['vol']

        #Define trade specific dataframes
        specific_df_start_time = time.time()

        @njit
        def find_next_initiator_numba(times, prices, initiators):
            n = len(times)

            tNextSell_tob = np.full(n, np.nan)
            pNextSell_tob = np.full(n, np.nan)
            tNextBuy_tos = np.full(n, np.nan)
            pNextBuy_tos = np.full(n, np.nan)

            for i in range(n):
                if initiators[i] == 1:
                    for j in range(i + 1, n):
                        if initiators[j] == -1:
                            tNextSell_tob[i] = times[j]
                            pNextSell_tob[i] = prices[j]
                            break

            for i in range(n):
                if initiators[i] == -1:
                    for j in range(i + 1, n):
                        if initiators[j] == 1:
                            tNextBuy_tos[i] = times[j]
                            pNextBuy_tos[i] = prices[j]
                            break

            return tNextSell_tob, pNextSell_tob, tNextBuy_tos, pNextBuy_tos
        
        def find_next_initiator(df):
            times = df['time_float'].values
            prices = df['price'].values
            initiators = df['Initiator'].values

            tNextSell_tob, pNextSell_tob, tNextBuy_tos, pNextBuy_tos = find_next_initiator_numba(times, prices, initiators)

            df['tNextSell_tob'] = tNextSell_tob
            df['pNextSell_tob'] = pNextSell_tob
            df['tNextBuy_tos'] = tNextBuy_tos
            df['pNextBuy_tos'] = pNextBuy_tos

            return df

        trades = find_next_initiator(trades)

        #Define the Buys_trades and Sell_trades dataframes
        Buys_trades = trades[trades["Initiator"] == 1].copy()
        Sells_trades = trades[trades["Initiator"] == -1].copy()
        Buys_trades.drop(columns=['tNextBuy_tos', 'pNextBuy_tos'], inplace=True)
        Sells_trades.drop(columns=['tNextSell_tob', 'pNextSell_tob'], inplace=True)

        Buys_trades['dtNextSell_tob'] = Buys_trades['tNextSell_tob'] - Buys_trades['time_float']
        Sells_trades['dtNextBuy_tos'] = Sells_trades['tNextBuy_tos'] - Sells_trades['time_float']

        #Define the Retail_trades dataframe
        tradessigns_copy = tradessigns[tradessigns['EX'] == 'D'].copy()
        tradessigns_copy.drop(columns=['Initiator'], inplace=True)
        tradessigns_copy['supbenny'] = tradessigns_copy['price'] % 0.01 * 100
        tradessigns_copy = tradessigns_copy[tradessigns_copy['price'] != tradessigns_copy['midpoint']].copy()

        #If the trade is not inside the matching NBBO and the spread is 1 cent apply the older retail trades identification
        tradessigns_copy['correct_sign'] = tradessigns_copy['price'].between(tradessigns_copy['bid'], tradessigns_copy['ask']).copy()
        mask = (~(tradessigns_copy['correct_sign'])) & (tradessigns_copy['spread'] == 0.01)
        inverse_mask = ~mask

        #Old method of retail trades
        Retail_trades_old = tradessigns_copy[mask].copy()
        Retail_trades_old['trade_type'] = Retail_trades_old['supbenny'].apply(identify_retail_old)
        Retail_trades_old = Retail_trades_old[Retail_trades_old['trade_type'] == 'retail trade'].copy().drop(columns=['trade_type'])
        Retail_trades_old['Initiator'] = 0
        Retail_trades_old.loc[Retail_trades_old['supbenny'] < 0.04, 'Initiator'] = 1
        Retail_trades_old.loc[Retail_trades_old['supbenny'] > 0.06, 'Initiator'] = -1

        #New method of retail trades
        Retail_trades_new = tradessigns_copy[inverse_mask].copy()
        Retail_trades_new['trade_type'] = Retail_trades_new['supbenny'].apply(identify_retail)
        Retail_trades_new = Retail_trades_new[Retail_trades_new['trade_type'] == 'retail trade'].copy().drop(columns=['trade_type'])
        Retail_trades_new['lower_bound'] = Retail_trades_new['bid'] + 0.4 * Retail_trades_new['spread']
        Retail_trades_new['upper_bound'] =  Retail_trades_new['bid'] + 0.6 * Retail_trades_new['spread']
        Retail_trades_new['retail'] = ~Retail_trades_new['price'].between(Retail_trades_new['lower_bound'], Retail_trades_new['upper_bound'])
        Retail_trades_new = Retail_trades_new[(Retail_trades_new['retail'])].copy()
        Retail_trades_new['Initiator'] = 0
        Retail_trades_new.loc[Retail_trades_new['price'] > Retail_trades_new['midpoint'], 'Initiator'] = 1
        Retail_trades_new.loc[Retail_trades_new['price'] < Retail_trades_new['midpoint'], 'Initiator'] = -1
        Retail_trades_new = Retail_trades_new.drop(columns=['lower_bound', 'upper_bound', 'retail'])
        Retail_trades = pd.concat([Retail_trades_old, Retail_trades_new])
        Retail_trades.sort_values(by='time', inplace=True)

        #Discern Buys from Sells
        Retail_trades = find_next_initiator(Retail_trades)
        Buys_Retail_trades = Retail_trades[Retail_trades["Initiator"] == 1].copy()
        Sells_Retail_trades = Retail_trades[Retail_trades["Initiator"] == -1].copy()
        Buys_Retail_trades.drop(columns=['tNextBuy_tos', 'pNextBuy_tos'], inplace=True)
        Sells_Retail_trades.drop(columns=['tNextSell_tob', 'pNextSell_tob'], inplace=True)
        Buys_Retail_trades['dtNextSell_tob'] = Buys_Retail_trades['tNextSell_tob'] - Buys_Retail_trades['time_float']
        Sells_Retail_trades['dtNextBuy_tos'] = Sells_Retail_trades['tNextBuy_tos'] - Sells_Retail_trades['time_float']

        #Define the Oddlot_trades dataframe
        target_date = datetime(2013, 12, 9)
        Oddlot_trades = trades[(trades['time'] >= target_date) & (trades['cond'].str.contains("I"))].copy()
        #Discern Buys from Sells
        Oddlot_trades = find_next_initiator(Oddlot_trades)
        Buys_Oddlot_trades = Oddlot_trades[Oddlot_trades["Initiator"] == 1].copy()
        Sells_Oddlot_trades = Oddlot_trades[Oddlot_trades["Initiator"] == -1].copy()
        Buys_Oddlot_trades.drop(columns=['tNextBuy_tos', 'pNextBuy_tos'], inplace=True)
        Sells_Oddlot_trades.drop(columns=['tNextSell_tob', 'pNextSell_tob'], inplace=True)
        Buys_Oddlot_trades['dtNextSell_tob'] = Buys_Oddlot_trades['tNextSell_tob'] - Buys_Oddlot_trades['time_float']
        Sells_Oddlot_trades['dtNextBuy_tos'] = Sells_Oddlot_trades['tNextBuy_tos'] - Sells_Oddlot_trades['time_float']
        specific_df_end_time = time.time()
        
        #Define the Returns dataframe
        returns_start_time = time.time()
        trade_returns = calculate_returns_shift(trades, price_col='price', time_col='time', additional_cols=['vol'])
        midprice_returns = calculate_returns_shift(Midpoint, price_col='price', time_col='time')
        returns_end_time = time.time()

        #Define the trade_signs dataframe
        trade_signs = tradessigns[
            ["time", "Initiator", "vol"]].copy()
        trade_signs.rename(columns={"Initiator": "returns"}, inplace=True) #rename the sign as return for convenience, (for computing autocorrelation as on returns)

        #Reset indices to the new dataframes
        trades.reset_index(inplace = True)
        Buys_trades.reset_index(inplace = True)
        Sells_trades.reset_index(inplace = True)
        Retail_trades.reset_index(inplace = True)
        Buys_Retail_trades.reset_index(inplace = True)
        Sells_Retail_trades.reset_index(inplace = True)
        Oddlot_trades.reset_index(inplace = True)
        Buys_Oddlot_trades.reset_index(inplace = True)
        Sells_Oddlot_trades.reset_index(inplace = True)
        
        #Write the time analysis
        if prep_analysis_path is not None and stock_name == "IBM":
            with open(prep_analysis_path, "a") as f:
                f.write(f"Stock: {stock_name}\n")
                f.write(f"Suffix: {s}\n")
                f.write(f"Day: {base_date}\n")
                f.write(f"Load time: {load_time} seconds\n")
                f.write(f"decode time: {decode_end_time - decode_start_time} seconds\n")
                f.write(f"format time: {format_end_time - format_start_time} seconds\n")
                f.write(f"Clean time: {clean_end_time - clean_start_time - trsigns_time} seconds\n")
                f.write(f"specific trades time: {specific_df_end_time - specific_df_start_time} seconds\n")
                f.write(f"returns time: {returns_end_time - returns_start_time} seconds\n")
                f.write(f"TradeSigns time: {trsigns_time} seconds\n")

        return trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Buys_Oddlot_trades, Sells_Oddlot_trades, Buys_Retail_trades, Sells_Retail_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs

    except  NoTradesException:
        return None
    except  NoNbbosException:
        return None

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

