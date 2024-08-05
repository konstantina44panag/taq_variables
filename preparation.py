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
from decimal import Decimal, getcontext
getcontext().prec = 16

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
def load_trades_dataset(
    hdf_file,
    dataset_path,
    columns_of_interest,
    corr_pattern="CORR",
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
def load_quotes_dataset(
    hdf_file,
    dataset_path,
    columns_of_interest,
    cond_pattern="COND",
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
    epsilon = 1e-10
    if epsilon < z < 1 - epsilon :
        return 'retail trade'
    else:
        return 'non-retail trade'
    
def identify_retail_old(z):
    epsilon = 1e-10
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

dummy_data = np.random.rand(100)
_ = rolling_median_exclude_self(dummy_data, 5)
_ = rolling_mad_exclude_self(dummy_data, 5)

#def prepare_datasets: Contains the loading of data to dataframes and applies the appropriate operations, this function is called by the python script variables_v4.py which then calculates variables
def prepare_datasets(hdf5_file_path, base_date, stock_name, year, month, day, method, freq, ctm_dataset_path, complete_nbbo_dataset_path, hdf5_variable_path, prep_analysis_path=None, emp_analysis_path=None, var_analysis_path=None, prof_analysis_path=None):
    try:
        load_start_time = time.time()

        # Load datasets
        with tables.open_file(hdf5_file_path, "r") as hdf:
            trades = load_trades_dataset(
                hdf,
                ctm_dataset_path,
                ["TIME_M", "EX", "PRICE", "SIZE"],
                corr_pattern="CORR",
                cond_pattern="COND",
            )
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
       
        #Remove nan or zero prices, P2 cleaning step in Taq Cleaning Techniques google doc 
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
            
        #Cleaning step T3
        pl_trades = handle_duplicates(pl_trades, key_col=['datetime'], value_cols=['price'], sum_col=['vol'], other_cols=['time', "corr"], join_col=['cond', "EX"])

        #switch to pandas
        trades = pl_trades.to_pandas()
        trades.reset_index(drop=True)
      
        #Cleaning nbbos
        #Remove nan or zero quotes, cleaning step P2*
        mask = (~np.isnan(nbbos['BEST_ASK'].values)) & (~np.isnan(nbbos['BEST_BID'].values)) & (nbbos['BEST_ASK'].values != 0) & (nbbos['BEST_BID'].values != 0)
        nbbos = nbbos.loc[mask]        
        #H&j
        pl_nbbos = pl.from_pandas(nbbos)
        #mask1 = (pl.col("BEST_ASK") <= 0) & (pl.col("BEST_BID") <= 0)
        #mask2 = (pl.col("Best_AskSizeShares") <= 0) & (pl.col("Best_BidSizeShares") <= 0)
        #mask3 = pl.col("BEST_ASK").is_null() & pl.col("BEST_BID").is_null()
        #mask4 = pl.col("Best_AskSizeShares").is_null() & pl.col("Best_BidSizeShares").is_null()

        # Combine masks
        #combined_mask = ~(mask1 | mask2 | mask3 | mask4)
        #pl_nbbos = pl_nbbos.filter(combined_mask)
        
        #Check for empty dataframe after the cleaning step
        if pl_nbbos.height == 0:
            print(f"No nbbos after cleaning techniques for {stock_name}")
            raise NoNbbosException()
        
        #Cleaning Step Q1
        pl_nbbos = handle_duplicates(pl_nbbos, key_col='datetime', value_cols=['BEST_ASK', 'BEST_BID'],  sum_col=['Best_AskSizeShares', 'Best_BidSizeShares'], other_cols=['time', 'qu_cond'])

        #Cleaning Step Q2
        pl_nbbos = pl_nbbos.filter(pl_nbbos['BEST_ASK'] > pl_nbbos['BEST_BID'])
        
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
        #H&J
        #pl_nbbos = pl_nbbos.with_columns(
        #    [
        #        pl.col("Midpoint").shift(1).alias("lmid")
        #    ]
        #)
        #pl_nbbos = pl_nbbos.with_columns(
        #    [
        #        pl.when(pl.col("lmid").is_null())
        #        .then(None).otherwise(pl.col("lmid")).alias("lmid")
        #    ]
        #)
        #pl_nbbos = pl_nbbos.with_columns(
        #    [
        #        (pl.col("lmid") - 2.5).alias("lm25"),
        #        (pl.col("lmid") + 2.5).alias("lp25")
        #    ]
        #)

        #switch to pandas
        nbbos = pl_nbbos.to_pandas()
        nbbos.reset_index(drop=True)

        #Cleaning Step Q4
        nbbos['rolling_median'] = rolling_median_exclude_self(nbbos['midpoint'].values, 50)
        nbbos['rolling_mad'] = rolling_mad_exclude_self(nbbos['midpoint'].values, 50)
        nbbos['exclude'] = np.abs(nbbos['midpoint'] - nbbos['rolling_median']) > 10 * nbbos['rolling_mad']
        nbbos = nbbos[~nbbos['exclude']]
        nbbos = nbbos.drop(columns=['rolling_median', 'rolling_mad', 'exclude'])
        
        #Check for empty dataframe after the cleaning step
        if nbbos.empty:
            print(f"No nbbos after cleaning techniques for {stock_name}")
            raise NoNbbosException()

        clean_only_end_time = time.time()
        
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
     
        #Define the Midpoint
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

        #sort dataframes and rename columns after trade signing
        Ask.sort_values(by="datetime", inplace=True)
        Bid.sort_values(by="datetime", inplace=True)
        Midpoint.sort_values(by="time", inplace=True)
        Ask.drop(columns=["time"], inplace=True)
        Ask.rename(columns={"datetime": "time"}, inplace=True)
        Bid.drop(columns=["time"], inplace=True)
        Bid.rename(columns={"datetime": "time"}, inplace=True)

        tradessigns.sort_values(by='datetime', inplace=True)
        tradessigns.rename(columns={"time": "time_float"}, inplace=True)
        tradessigns.rename(columns={"datetime": "time"}, inplace=True)

        #Now that trades are matched with a quote pair, apply cleaning step T4:
        tradessigns['spread'] = tradessigns['ask'] - tradessigns['bid']
        tradessigns['upper_bound'] = tradessigns['ask'] + tradessigns['spread']
        tradessigns['lower_bound'] = tradessigns['bid'] - tradessigns['spread']
        tradessigns = tradessigns[
            (tradessigns['price'] <= tradessigns['upper_bound']) &
            (tradessigns['price'] >= tradessigns['lower_bound'])
        ]
        tradessigns = tradessigns.drop(columns=['upper_bound', 'lower_bound'])
        
        if tradessigns.empty:
            print(f"No trades after cleaning techniques for {stock_name}")
            raise NoTradesException()
        
        tradessigns.sort_values(by='time', inplace=True)

        #Set trades to be the extended result of the trade signing, including the Initiator column
        trades = tradessigns

        #Define trade specific dataframes
        specific_df_start_time = time.time()

        @njit
        def find_next_initiator_numba(times, prices, initiators):
            n = len(times)

            tNextSell_tob = np.full(n, 0.00)
            pNextSell_tob = np.full(n, 0.00)
            tNextBuy_tos = np.full(n, 0.00)
            pNextBuy_tos = np.full(n, 0.00)

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
        tradessigns_copy = tradessigns_copy[tradessigns_copy['price'] != tradessigns_copy['midpoint']]

        #If the trade is not inside the matching NBBO and the spread is 1 cent apply the older retail trades identification
        tradessigns_copy['correct_sign'] = tradessigns_copy['price'].between(tradessigns_copy['bid'], tradessigns_copy['ask'])
        mask = (~(tradessigns_copy['correct_sign'])) & (tradessigns_copy['spread'] == 0.01)
        inverse_mask = ~mask

        #Old method of retail trades
        Retail_trades_old = tradessigns_copy[mask].copy()
        Retail_trades_old['trade_type'] = Retail_trades_old['supbenny'].apply(identify_retail_old)
        Retail_trades_old = Retail_trades_old[Retail_trades_old['trade_type'] == 'retail trade'].drop(columns=['trade_type'])
        Retail_trades_old['Initiator'] = 0
        Retail_trades_old.loc[Retail_trades_old['supbenny'] < 0.04, 'Initiator'] = 1
        Retail_trades_old.loc[Retail_trades_old['supbenny'] > 0.06, 'Initiator'] = -1

        #New method of retail trades
        Retail_trades_new = tradessigns_copy[inverse_mask].copy()
        Retail_trades_new['trade_type'] = Retail_trades_new['supbenny'].apply(identify_retail)
        Retail_trades_new = Retail_trades_new[Retail_trades_new['trade_type'] == 'retail trade'].drop(columns=['trade_type'])
        Retail_trades_new['lower_bound'] = Retail_trades_new['bid'] + 0.4 * Retail_trades_new['spread']
        Retail_trades_new['upper_bound'] =  Retail_trades_new['bid'] + 0.6 * Retail_trades_new['spread']
        Retail_trades_new['retail'] = ~Retail_trades_new['price'].between(Retail_trades_new['lower_bound'], Retail_trades_new['upper_bound'])
        Retail_trades_new = Retail_trades_new[(Retail_trades_new['retail'])]
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
        Oddlot_trades = tradessigns[(tradessigns['time'] >= target_date) & (tradessigns['cond'].str.contains("I"))].copy()

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
        trade_signs.rename(columns={"Initiator": "returns"}, inplace=True)

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
        if prep_analysis_path is not None and args.stock_name == "IBM":
            with open(prep_analysis_path, "a") as f:
                f.write(f"Stock: {stock_name}\n")
                f.write(f"Day: {base_date}\n")
                f.write(f"Load time: {load_time} seconds\n")
                f.write(f"decode time: {decode_end_time - decode_start_time} seconds\n")
                f.write(f"format time: {format_end_time - format_start_time} seconds\n")
                f.write(f"Clean time: {clean_only_end_time - clean_only_start_time} seconds\n")
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

