#!/usr/bin/env python3.11
"""This script computes important variables but first passes arguments to preparation.py."""

import pandas as pd
import numpy as np
import argparse
import cProfile
import pstats
import time
import polars as pl
from datetime import timedelta
from datetime import datetime
from statsmodels.tsa.stattools import acf
from preparation import prepare_datasets, NoTradesException, NoNbbosException
pd.set_option('display.max_rows', 300)

class NoTradesException(Exception):
    pass

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
parser.add_argument(
    "hdf5_variable_path",
    type=str,
    help="The path and name of the output variable file",
)
parser.add_argument(
    "--prep_analysis_path",
    type=str,
    help="The path and name of the preparation analysis file (optional)",
)
parser.add_argument(
    "emp_analysis_path",
    type=str,
    help="The path and name of the empty variables file",
)
parser.add_argument(
    "--var_analysis_path",
    type=str,
    help="The path and name of the variables analysis file (optional)",
)
parser.add_argument(
    "--prof_analysis_path",
    type=str,
    help="The path and name of the profiling analysis file (optional)",
)
args, unknown = parser.parse_known_args()

def main():
    global aggregated_data
    aggregated_data = {}
    aggregated_data_outside_trading = {}

    #Import datasets from preparation.py
    try:
        result = prepare_datasets(
            args.hdf5_file_path,
            args.base_date,
            args.stock_name,
            args.year,
            args.month,
            args.day,
            args.ctm_dataset_path,
            args.complete_nbbo_dataset_path,
            args.hdf5_variable_path,
            args.prep_analysis_path,
            args.emp_analysis_path,
            args.var_analysis_path,
            args.prof_analysis_path)

        if result is None:
            print(f"No trades to process for {args.stock_name} on {args.base_date}. Skipping further calculations.")
            return

        trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Buys_Oddlot_trades, Sells_Oddlot_trades, Buys_Retail_trades, Sells_Retail_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs = result

    except NoTradesException:
        print(f"No trades to process for {args.stock_name} on {args.base_date}. Skipping further calculations.")
        return
    except NoNbbosException:
        print(f"No NBBOs to process for {args.stock_name} on {args.base_date}. Skipping further calculations.")
        return
    
    except Exception as e:
        print(f"An error occurred while preparing datasets: {e}")
        return
    
    # Start timing the main calculations
    main_start_time = time.time()

    #Customized Functions for computing variables

    #Find Opne-Close prices
    def auction_conditions(df):
        pl_df = pl.from_pandas(df)
        special_conditions_df = pl_df.filter(pl.col('cond').str.contains('M|O|P|Q|5|6|9'))     #these codes correspond to opening and closing prices 
        cleaned_df = pl_df.filter(~pl.col('cond').str.contains('M|O|P|Q|5|6|9'))               #now remove opening and closing prices from trades dataframe
        special_conditions_pd = special_conditions_df.select(['time', 'price', 'vol', 'EX', 'cond']).to_pandas()
        cleaned_df_pd = cleaned_df.to_pandas()
        return special_conditions_pd, cleaned_df_pd
    
    #Calculate the variance
    def calculate_minute_variance(returns):
        n = len(returns)
        if n <= 1:
            return np.nan
        return returns.var()

    #Calculate the volatility
    def calculate_minute_volatility(returns):
        n = len(returns)
        if n <= 1:
            return np.nan
        return returns.std()

    #Calculate the autocorrelation
    def calculate_autocorrelation(returns, lag=1):
        x = returns.to_numpy()
        x = x[~np.isnan(x)]
        if len(x) <= lag:
            return np.nan
        if np.var(x) == 0 or np.var(x[:-lag]) == 0 or np.var(x[lag:]) == 0:
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    
    #Calculate the orderflow by Chordia, Hu, Subrahmanyam and Tong, MS 2019
    def calculate_oib_shr(df1, df2):
        if df1 is None or df1.empty or df1.isna().all().all() or df2 is None or df2.empty or df2.isna().all().all():
            return None
        
        if 'time' in df1.columns:
            df1.set_index('time', inplace=True)
        if 'time' in df2.columns:
            df2.set_index('time', inplace=True)

        df1_filtered = df1.between_time("09:30", "15:59:59").copy()
        df2_filtered = df2.between_time("09:30", "15:59:59").copy()
        
        if df1_filtered.empty or df2_filtered.empty:
            return None
        buys_per_s = df1_filtered.resample("s")["vol"].sum()
        sells_per_s = df2_filtered.resample("s")["vol"].sum()
        
        oib_shr_s = (buys_per_s - sells_per_s) / (buys_per_s + sells_per_s)

        return oib_shr_s

    #Calculate the variance and autocorrelation of orderflow by Chordia, Hu, Subrahmanyam and Tong, MS 2019
    def apply_voib_shr_aggregations(df):
        if df is None or df.empty or df.isna().all().all():
            return None
        if df.shape[0] == 1:
            return None
        pl_df = pl.from_pandas(df)

        resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left').agg([
        pl.col('OIB_SHR').map_elements(calculate_minute_volatility, return_dtype=pl.Float64).alias('OIB_volatility_s'),
        pl.col('OIB_SHR').map_elements(calculate_autocorrelation, return_dtype=pl.Float64).alias('OIB_autocorr_s'),
        ])
        return resampled_df.to_pandas().set_index('time')
            
    #Calculate the variance and autocorrelation of returns      
    def apply_return_aggregations(pl_df, column='returns', df_name=None):
        if pl_df is None or pl_df.shape[0] == 0:
            return None
        if pl_df.shape[0] == 1:
            return None
        if df_name is None:
            volatility_col_name = 'ret_volatility_s'
            autocorr_col_name = 'ret_autocorr_s'
        else:
            volatility_col_name = f'ret_volatility_{df_name}_s'
            autocorr_col_name = f'ret_autocorr_{df_name}_s'

        resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left').agg([
            pl.col(column).map_elements(calculate_minute_volatility, return_dtype=pl.Float64).alias(volatility_col_name),
            pl.col(column).map_elements(calculate_autocorrelation, return_dtype=pl.Float64).alias(autocorr_col_name),
        ])
        return resampled_df.to_pandas().set_index('time')
        
    #Calculate the variance and autocorrelation of returns outside trading hours          
    def apply_return_aggregations_outside_trading(pl_df, column='returns', df_name=None):
        if pl_df is None or pl_df.shape[0] == 0:
            return None
        if pl_df.shape[0] == 1:
            return None
        if df_name is None:
            volatility_col_name = 'ret_volatility_s'
            autocorr_col_name = 'ret_autocorr_s'
        else:
            volatility_col_name = f'ret_volatility_{df_name}_s'
            autocorr_col_name = f'ret_autocorr_{df_name}_s'

        resampled_df = pl_df.group_by_dynamic('time', every='30m', closed='left').agg([
            pl.col(column).map_elements(calculate_minute_volatility, return_dtype=pl.Float64).alias(volatility_col_name),
            pl.col(column).map_elements(calculate_autocorrelation, return_dtype=pl.Float64).alias(autocorr_col_name),
        ])
        return resampled_df.to_pandas().set_index('time')

     #Calculate only the variance of returns
    def apply_ret_variances_aggregations(pl_df, column='returns'):
        if pl_df is None or pl_df.shape[0] == 0:
            return None
        if pl_df.shape[0] == 1:
            return None
        resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left').agg([
        pl.col(column).map_elements(calculate_minute_variance, return_dtype=pl.Float64).alias('variance')])
           
        return resampled_df.to_pandas().set_index('time')
    
    
    #Fill the minutes in the time column after the time bars are calculated for different dataframes and variables, so that entire time bar data can be merged in one dataframe
    def reindex_to_full_time(df, base_date, outside_trading=False):
        if df is None or df.empty or df.isna().all().all():
            return None
        
        if outside_trading:
            morning_index = pd.date_range(start=f"{base_date} 03:30", end=f"{base_date} 09:29", freq="30min")
            evening_index = pd.date_range(start=f"{base_date} 16:00", end=f"{base_date} 20:00", freq="30min")
            full_time_index = morning_index.union(evening_index)
        else:
            full_time_index = pd.date_range(start=f"{base_date} 09:30", end=f"{base_date} 15:59", freq="1min")
        df_reindexed = df.reindex(full_time_index)
        return df_reindexed

    # Aggregated Functions for Trades
    #Variables for the trades dataframes, most of them are formed at the same time 
    def apply_aggregations(df_filtered, df_name):
        #Check for empty dataframe or for dimension size 1 when there is no reason for creating minute bars
        if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
            return None
        if len(df_filtered) == 1:
            return df_filtered
           
        #Make sure the time column is a column and not an index
        df_filtered.reset_index(inplace = True)
        
        #Calculate durations of prices and weighted_price by these durations
        df_filtered['durations'] = df_filtered['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        df_filtered['weighted_price'] = df_filtered['price'] * df_filtered['durations']

        #Convert to polars
        pl_df = pl.from_pandas(df_filtered)

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='1m', label='left').agg([
                pl.col('count').max().alias('max_events_s')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()
                
            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))
                
            #Unite the different variables/aggregation methods, to be computed at the same time
            aggregations = [
                pl.col('price').last().alias('last_price'),
                pl.col('vol').last().alias('last_vol'),
                pl.col('time').last().alias('last_time'),
                pl.col('price').mean().alias('avg_price'),
                pl.col('vol').mean().alias('avg_vol'),
                pl.col('vol').sum().alias('tot_vol'),
                calculate_vwap_pl().alias('vwap'),
                calculate_twap_pl().alias('twap'),
                pl.count('price').alias('num_events')
            ]
            if df_name == 'Buys_trades':
                aggregations.append(pl.col('pNextSell_tob').mean().alias('pNextSell_avg'))
                aggregations.append(pl.col('dtNextSell_tob').mean().alias('dtNextSell_avg'))

            if df_name == 'Sells_trades':
                aggregations.append(pl.col('pNextBuy_tos').mean().alias('pNextBuy_avg'))
                aggregations.append(pl.col('dtNextBuy_tos').mean().alias('dtNextBuy_avg'))

            #Resample to one minute bars, using the aggregations above
            resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg(aggregations)

            #Merge with the aggregation of maximum number of trades which was performed prior to the others
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            
            #Return the time bar variables, set the time as an index
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
   
   #Variables for the trades dataframes, almost the same operations as the def apply_aggregations, but this function is used for outside trading hours. Resampling in 30 minutes!
    def apply_aggregations_outside_trading(df_filtered, df_name):
        if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
            return None
        if len(df_filtered) == 1:
            return df_filtered
       
        df_filtered.reset_index(inplace = True)
        
        #Calculate durations of prices and weighted_price by these durations
        df_filtered['durations'] = df_filtered['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        df_filtered['weighted_price'] = df_filtered['price'] * df_filtered['durations']

        pl_df = pl.from_pandas(df_filtered)

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])
            
            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='30m', label='left').agg([
                pl.col('count').max().alias('max_events_s')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()

            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))
            
            aggregations = [
                pl.col('price').last().alias('last_price'),
                pl.col('vol').last().alias('last_vol'),
                pl.col('time').last().alias('last_time'),
                pl.col('price').mean().alias('avg_price'),
                pl.col('vol').mean().alias('avg_vol'),
                pl.col('vol').sum().alias('tot_vol'),
                calculate_vwap_pl().alias('vwap'),
                calculate_twap_pl().alias('twap'),
                pl.count('price').alias('num_events')
            ]

            if df_name == 'Buys_trades':
                aggregations.append(pl.col('pNextSell_tob').mean().alias('pNextSell_avg'))
                aggregations.append(pl.col('dtNextSell_tob').mean().alias('dtNextSell_avg'))

            if df_name == 'Sells_trades':
                aggregations.append(pl.col('pNextBuy_tos').mean().alias('pNextBuy_avg'))
                aggregations.append(pl.col('dtNextBuy_tos').mean().alias('dtNextBuy_avg'))
                
            resampled_df = pl_df.group_by_dynamic('time', every='30m', closed='left', label='left').agg(aggregations)

            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    #Aggregated Functions for Quotes
    def apply_quote_aggregations(df_filtered, df_name):
        if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
            return None
        if len(df_filtered) == 1:
            return df_filtered

        df_filtered.reset_index(inplace = True)

        #Calculate durations of prices and weighted_price by these durations
        df_filtered['durations'] = df_filtered['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        df_filtered['weighted_price'] = df_filtered['price'] * df_filtered['durations']

        pl_df = pl.from_pandas(df_filtered)

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='1m', label='left').agg([
                pl.col('count').max().alias('max_events_s')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()

            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))
                
            #Find the trading halts, or news event indicator
            def encode_conditions_expr():
                valid_chars = 'DIJKLMNOPQRSTVYZ124'
                return pl.col('qu_cond').map_elements(
                    lambda x: ''.join(sorted(set([c for c in x if c in valid_chars]))), return_dtype=pl.Utf8
                )

            
            aggregations = [
                pl.col('price').last().alias('last_price'),
                pl.col('vol').last().alias('last_vol'),
                pl.col('time').last().alias('last_time'),
                pl.col('price').mean().alias('avg_price'),
                pl.col('vol').mean().alias('avg_vol'),
                pl.col('vol').sum().alias('tot_vol'),
                calculate_vwap_pl().alias('vwap'),
                calculate_twap_pl().alias('twap'),
                pl.count('price').alias('num_events'),
                encode_conditions_expr().alias('halt_indic')
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def apply_quote_aggregations_outside_trading(df_filtered, df_name):
        if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
            return None
        if len(df_filtered) == 1:
            return df_filtered
        
        df_filtered.reset_index(inplace = True)
        
        #Calculate durations of prices and weighted_price by these durations
        df_filtered['durations'] = df_filtered['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        df_filtered['weighted_price'] = df_filtered['price'] * df_filtered['durations']

        pl_df = pl.from_pandas(df_filtered)

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='30m', label='left').agg([
                pl.col('count').max().alias('max_events_s')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()

            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))

            #Find the trading halts, or news event indicator
            def encode_conditions_expr():
                valid_chars = 'DIJKLMNOPQRSTVYZ124'
                return pl.col('qu_cond').map_elements(
                    lambda x: ''.join(sorted(set([c for c in x if c in valid_chars]))), return_dtype=pl.Utf8
                )
            
            aggregations = [
                pl.col('price').last().alias('last_price'),
                pl.col('vol').last().alias('last_vol'),
                pl.col('time').last().alias('last_time'),
                pl.col('price').mean().alias('avg_price'),
                pl.col('vol').mean().alias('avg_vol'),
                pl.col('vol').sum().alias('tot_vol'),
                calculate_vwap_pl().alias('vwap'),
                calculate_twap_pl().alias('twap'),
                pl.count('price').alias('num_events'),
                encode_conditions_expr().alias('halt_indic')
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='30m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    #Functions for Midprice

    def apply_midpoint_aggregations(df_filtered):
        if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
            return None
        if len(df_filtered) == 1:
            return df_filtered
        
        pl_df = pl.from_pandas(df_filtered.reset_index())

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='1m', label='left').agg([
                pl.col('count').max().alias('max_events_s')
            ])

            #Count the number of events
            aggregations = [
                pl.count('price').alias('num_events'),
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return
        


    def apply_midpoint_aggregations_outside_trading(df_filtered):
        if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
            return None
        if len(df_filtered) == 1:
            return df_filtered

        pl_df = pl.from_pandas(df_filtered.reset_index())

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='30m', label='left').agg([
                pl.col('count').max().alias('max_events_s')
            ])

            #Count the number of events
            aggregations = [
                pl.count('price').alias('num_events'),
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='30m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return


    def process_resample_data(df, interval, base_date=None, outside_trading=False):
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df

        if 'time' in df.columns:
            df.set_index('time', inplace=True)

        #Filter for the appropriate hours
        if outside_trading:
            start_time_morning = f"{base_date} 09:30"
            end_time_afternoon = f"{base_date} 16:00"
            df_filtered = df[(df.index < start_time_morning) | (df.index >= end_time_afternoon)].copy()
        else:
            df_filtered = df.between_time("09:30", "15:59:59").copy()
            
        #Check for an empty filtered dataframe
        if df_filtered.empty or df_filtered.isna().all().all():
            return None
            
        #Reset the time column from index and convert to polars
        df_filtered = df_filtered.reset_index()
        pl_df = pl.from_pandas(df_filtered)

        #Compute the Volume Weighted Return to be used for second bars
        def calculate_vwapr_expr():
            return (pl.col('returns') * pl.col('vol')).sum() / pl.col('vol').sum()

        #Compute the mean Return to be used for second bars, as an alternative for the Volume Weighted Return, e.g. for the midpoint returns where the volume column is not defined 
        def calculate_mean_return_expr():
            return pl.col('returns').mean()

        aggregations = [
            calculate_vwapr_expr().alias('returns'),
        ] if 'vol' in df.columns else [
            calculate_mean_return_expr().alias('returns')
        ]

        #Resample on the interval, which is an input argument of the function
        resampled_df = pl_df.group_by_dynamic('time', every=interval, closed='left').agg(aggregations)

        if resampled_df.is_empty():
            return None
    
        #since the volume has been used for the weighting, it is now dropped from the dataframe
        if 'vol' in resampled_df.columns:
            resampled_df = resampled_df.drop('vol')
        
        return resampled_df
    
    def process_daily(interval, df_interval, cond_char_set, is_cond=True):
        def calculate_vwap(df):
            if df.height == 0:
                return 0.0
            return (df['price'] * df['vol']).sum() / df['vol'].sum()

        # Dictionaries to store the results
        daily_inside = {}
        daily_outside = {}

        # Loop through both intervals: inside and outside
        for interval_name, df_interval in zip(['inside', 'outside'], [df_filtered_inside, df_filtered_outside]):
        
            # Loop through characters in the set (condition or exchange)
            for char in cond_char_set:
                if char == '@':
                    key = 'cond_at' if is_cond else 'vwap_ex_at'
                elif char == '':
                    key = 'cond_empty' if is_cond else 'vwap_ex_empty'
                else:
                    key = f'cond_{char}' if is_cond else f'vwap_ex_{char}'

                if char == '' and is_cond:
                    df_filtered = df_interval.filter(pl.col('cond') == '')
                else:
                    df_filtered = df_interval.filter(pl.col('cond').str.contains(char))

                vwap = calculate_vwap(df_filtered)
                tot_vol = df_filtered['vol'].sum()
                no_buys = df_filtered.filter(pl.col('Initiator') == 1).height
                no_sells = df_filtered.filter(pl.col('Initiator') == -1).height

                if interval_name == 'inside':
                    if key not in daily_inside:
                        daily_inside[key] = {}
                    daily_inside[key]['vwap'] = vwap
                    daily_inside[key]['tot_vol'] = tot_vol
                    daily_inside[key]['no_buys'] = no_buys
                    daily_inside[key]['no_sells'] = no_sells
                else:
                    if key not in daily_outside:
                        daily_outside[key] = {}
                    daily_outside[key]['vwap'] = vwap
                    daily_outside[key]['tot_vol'] = tot_vol
                    daily_outside[key]['no_buys'] = no_buys
                    daily_outside[key]['no_sells'] = no_sells
            if is_cond == False:
                if interval_name == 'inside':
                    daily_inside['tot_vol'] = df_interval['vol'].sum()
                    daily_inside['no_buys'] = df_interval.filter(pl.col('Initiator') == 1).height
                    daily_inside['no_sells'] = df_interval.filter(pl.col('Initiator') == -1).height
                else:
                    daily_outside['total_vol'] = df_interval['vol'].sum()
                    daily_outside['no_buys'] = df_interval.filter(pl.col('Initiator') == 1).height
                    daily_outside['no_sells'] = df_interval.filter(pl.col('Initiator') == -1).height

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_daily_inside = flatten_dict(daily_inside)
        flat_daily_outside = flatten_dict(daily_outside)
        
        daily_inside_df = pd.DataFrame([flat_daily_inside])
        daily_outside_df = pd.DataFrame([flat_daily_outside])
        
        return daily_inside_df, daily_outside_df
    

    #End of function definitions, computations follow
   
    #Process daily bars
    exchanges_set = list('ABCDIJKMNPSTQVWXYZ')
    conditions_set = [''] + list('@ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')

    if 'time' in trades.columns:
            trades.set_index('time', inplace=True)

    df_filtered_inside  = trades.between_time("09:30", "15:59:59").copy()
    start_time_morning = f"{args.base_date} 09:30"
    end_time_afternoon = f"{args.base_date} 16:00"
    df_filtered_outside = trades[(trades.index < start_time_morning) | (trades.index >= end_time_afternoon)].copy()

    trades.reset_index(inplace=True)
    df_filtered_inside =  pl.from_pandas(df_filtered_inside)
    df_filtered_outside = pl.from_pandas(df_filtered_outside)

    daily_inside_df_cond, daily_outside_df_cond = process_daily('inside', df_filtered_inside, conditions_set, is_cond=True)
    daily_inside_df_ex, daily_outside_df_ex = process_daily('inside', df_filtered_inside, exchanges_set, is_cond=False)

    daily_inside_df = pd.concat([daily_inside_df_cond, daily_inside_df_ex], axis=1)
    daily_outside_df = pd.concat([daily_outside_df_cond, daily_outside_df_ex], axis=1)

    #Cleaning step T2, clear trade conditions after the computation of daily bars
    
    df = pl.from_pandas(trades)
    df = df.filter(~df['cond'].str.contains('B|G|J|K|L|W|Z'))
    trades = df.to_pandas()

    #Check for empty dataframe after the cleaning step
    if trades.empty:
        raise NoTradesException(f"No trades after cleaning techniques for {args.stock_name}")

    #Processing open /close auction prices
    start_auction_time = time.time()
    auction_conditions_df, trades = auction_conditions(trades)
    end_auction_time = time.time()

    #Process 1-minute bars
    # Processing Trades
    #For every trade dataframe, apply the function apply_aggregations from above
    start_process_trades_time = time.time()

    trade_dataframes_to_process = {
        "trades": trades,
        "Buys_trades": Buys_trades,
        "Sells_trades": Sells_trades,
        "Retail_trades": Retail_trades,
        "Buys_Retail_trades": Buys_Retail_trades,
        "Sells_Retail_trades": Sells_Retail_trades,
        "Oddlot_trades": Oddlot_trades,
        "Buys_Oddlot_trades": Buys_Oddlot_trades,
        "Sells_Oddlot_trades": Sells_Oddlot_trades
    }

    for name, df in trade_dataframes_to_process.items():
        if df is None or df.empty or df.isna().all().all():
            continue
         
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
      
        df_filtered = df.between_time("09:30", "15:59:59").copy()
        start_time_morning = f"{args.base_date} 09:30"
        end_time_afternoon = f"{args.base_date} 16:00"
        df_filtered_outside = df[(df.index < start_time_morning) | (df.index >= end_time_afternoon)].copy()
 
        if not df_filtered.empty:
            try:
                print(f"Processing {name} DataFrame")
                agg_df = apply_aggregations(df_filtered, name)
                aggregated_data[name] = reindex_to_full_time(agg_df, args.base_date)                
            except KeyError as e:
                print(f"Error processing {name}: {e}")
                continue

        if not df_filtered_outside.empty:
            try:
                print(f"Processing {name} DataFrame outside trading hours")
                agg_df_outside_trading = apply_aggregations_outside_trading(df_filtered_outside, name)
                aggregated_data_outside_trading[name] = reindex_to_full_time(agg_df_outside_trading, args.base_date, outside_trading=True)
            except KeyError as e:
                print(f"Error processing {name}: {e} outside trading hours")
                continue

    end_process_trades_time = time.time()

    #Extra variables for trades
    
    #The orderflow calculation is performed seperately
    start_process_OIB_time = time.time()
    print(f"Processing OIB statistics")

    # orderflow estimation from Buys and Sells
    trade_pairs = [
        (Buys_trades, Sells_trades, "Orderflow"),
        (Buys_Oddlot_trades, Sells_Oddlot_trades, "Orderflow_Oddlot"),
        (Buys_Retail_trades, Sells_Retail_trades, "Orderflow_Retail")

    ]

    for buys_trades, sells_trades, key in trade_pairs:
        # Calculate oib_shr
        oib_shr_s = calculate_oib_shr(buys_trades, sells_trades)
        if oib_shr_s is not None:
            oib_shr_df = oib_shr_s.to_frame(name='OIB_SHR').reset_index()
            oib_shr_df.columns = ['time', 'OIB_SHR']
        else:
            oib_shr_df = pd.DataFrame(columns=['time', 'OIB_SHR'])

        # Calculate the orderflow statistics and store in the dictionary
        aggregated_data[key] = reindex_to_full_time(apply_voib_shr_aggregations(oib_shr_df), args.base_date)

    end_process_OIB_time = time.time()

    #Herfindahl Index is performed seperately
    start_process_herfindahl_time = time.time()
    print(f"Processing Herfindahl Index")

    def calculate_hindex(df, name):
        if df is None or df.empty or df.isna().all().all():
            return None
        if 'time' in df.columns:
            df.set_index('time', inplace=True)

        df_filtered = df.between_time("09:30", "15:59:59").copy()
        if df_filtered.empty or df_filtered.isna().all().all():
            return None
        
        pl_df = pl.from_pandas(df_filtered.reset_index())
        
        resampled = pl_df.group_by_dynamic('time', every='1s').agg([
            pl.col('value').sum().alias('sum_of_values'),
            (pl.col('value')**2).sum().alias('sum_of_squared_values')
        ])
        
        minutely_data = resampled.group_by_dynamic('time', every='1m').agg([
            pl.col('sum_of_values').sum(),
            pl.col('sum_of_squared_values').sum()
        ])
        minutely_data = minutely_data.with_columns([
            (minutely_data['sum_of_values']**2).alias('sum_of_values_squared')
        ])
        
        minutely_data = minutely_data.with_columns([
            (minutely_data['sum_of_squared_values'] / minutely_data['sum_of_values_squared']).alias('Herfindahl_s')
        ])
        minutely_data = minutely_data.select([
            'time', 'Herfindahl_s'
        ])
        aggregated_data[f"Herfindahl_{name}"] = reindex_to_full_time(minutely_data.to_pandas().set_index('time'), args.base_date)


    #Apply the Herfindahl Index for these dataframes
    for df, name in zip([trades, Ask, Bid], 
                        ["trades", "Ask", "Bid"]):
        calculate_hindex(df, name)
    
    end_process_herfindahl_time = time.time()

    #Processing Midpoint, apply the function apply_midpoint_aggregations from above
    start_process_midpoint_time = time.time()

    if not Midpoint.empty:

        if 'time' in Midpoint.columns:
            Midpoint.set_index('time', inplace=True)
      
        df_filtered = Midpoint.between_time("09:30", "15:59:59").copy()
        start_time_morning = f"{args.base_date} 09:30"
        end_time_afternoon = f"{args.base_date} 16:00"
        df_filtered_outside = Midpoint[(Midpoint.index < start_time_morning) | (Midpoint.index >= end_time_afternoon)].copy()
        
        if not df_filtered.empty:
            try:
                print(f"Processing Midpoint DataFrame")
                midpoint_agg_df = apply_midpoint_aggregations(df_filtered)
                aggregated_data["Midpoint"] = reindex_to_full_time(midpoint_agg_df, args.base_date)
            except KeyError as e:
                print(f"Error processing Midpoint: {e}")

        if not df_filtered_outside.empty:
            try:
                print(f"Processing Midpoint DataFrame outside trading hours")
                midpoint_agg_df_outside_trading = apply_midpoint_aggregations_outside_trading(df_filtered_outside)
                aggregated_data_outside_trading["Midpoint"] = reindex_to_full_time(midpoint_agg_df_outside_trading, args.base_date, outside_trading=True)
            except KeyError as e:
                print(f"Error processing Midpoint outside trading hours: {e}")        

    end_process_midpoint_time = time.time()

    #Processing Quotes, apply the function apply_quote_aggregations from above
    start_process_quotes_time = time.time()

    quote_dataframes_to_process = {
        "Ask": Ask,
        "Bid": Bid,
    }

    for name, df in quote_dataframes_to_process.items():
        if df is None or df.empty or df.isna().all().all():
            continue
            
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
      
        df_filtered = df.between_time("09:30", "15:59:59").copy()
        start_time_morning = f"{args.base_date} 09:30"
        end_time_afternoon = f"{args.base_date} 16:00"
        df_filtered_outside = df[(df.index < start_time_morning) | (df.index >= end_time_afternoon)].copy()

        if not df_filtered.empty:
            try:
                print(f"Processing {name} DataFrame")
                agg_df = apply_quote_aggregations(df_filtered, name)
                aggregated_data[name] = reindex_to_full_time(agg_df,  args.base_date)          
            except KeyError as e:
                print(f"Error processing {name}: {e}")
                continue

        if not df_filtered_outside.empty:
            try:
                print(f"Processing {name} DataFrame outside trading hours")
                agg_df_outside_trading = apply_quote_aggregations_outside_trading(df_filtered_outside, name)
                aggregated_data_outside_trading[name] = reindex_to_full_time(agg_df_outside_trading,  args.base_date, outside_trading=True)
            except KeyError as e:
                print(f"Error processing {name}: {e} outside trading hours")
                continue

    end_process_quotes_time = time.time()
    

    #Processing Returns
    print(f"Processing Returns")
    start_process_returns_time = time.time()

    #Create one second returns for the trade_returns (from trade prices),  for the midprice_returns (from quote midpoint), for the trade_signs and the nbbo_signs for inside and outside trading hours
    trade_returns_1s = process_resample_data(trade_returns, '1s', args.base_date)    
    midprice_returns_1s = process_resample_data(midprice_returns, '1s', args.base_date)
    trade_returns_1s_outside_trading = process_resample_data(trade_returns, '1s', args.base_date, outside_trading=True)
    midprice_returns_1s_outside_trading = process_resample_data(midprice_returns, '1s', args.base_date, outside_trading=True)
    nbbo_signs_1s = process_resample_data(nbbo_signs, '1s', args.base_date)
    trade_signs_1s = process_resample_data(trade_signs, '1s', args.base_date)
    nbbo_signs_1s_outside_trading = process_resample_data(nbbo_signs, '1s', args.base_date, outside_trading=True)
    trade_signs_1s_outside_trading = process_resample_data(trade_signs, '1s', args.base_date, outside_trading=True)

    #Create one minute timebars with statistics of the one second returns, such as variance and autocorrelation, for inside and outside trading hours
    aggregated_data["trade_returns"] = reindex_to_full_time(apply_return_aggregations(trade_returns_1s, column='returns'), args.base_date)
    aggregated_data["midprice_returns"] = reindex_to_full_time(apply_return_aggregations(midprice_returns_1s, column='returns'),  args.base_date)
    aggregated_data["nbbo_sign_stat"] = reindex_to_full_time(apply_return_aggregations(nbbo_signs_1s, column='returns', df_name='nbbo_sign'),  args.base_date)
    aggregated_data["trade_sign_stat"] = reindex_to_full_time(apply_return_aggregations(trade_signs_1s, column='returns', df_name='trade_sign'),  args.base_date)
    aggregated_data_outside_trading["trade_returns"] = reindex_to_full_time(apply_return_aggregations_outside_trading(trade_returns_1s_outside_trading, column='returns', df_name='trade_ret'),  args.base_date, outside_trading=True)
    aggregated_data_outside_trading["midprice_returns"] = reindex_to_full_time(apply_return_aggregations_outside_trading(midprice_returns_1s_outside_trading, column='returns', df_name='midprice_ret'),  args.base_date, outside_trading=True)
    aggregated_data_outside_trading["nbbo_sign_stat"] = reindex_to_full_time(apply_return_aggregations_outside_trading(nbbo_signs_1s_outside_trading, column='returns', df_name='nbbo_sign'),  args.base_date, outside_trading=True)
    aggregated_data_outside_trading["trade_sign_stat"] = reindex_to_full_time(apply_return_aggregations_outside_trading(trade_signs_1s_outside_trading, column='returns', df_name='trade_sign'),  args.base_date, outside_trading=True)
    end_process_returns_time = time.time()

    #Variance Ratios
    start_process_vr_returns_time = time.time()
    
    for returns_df in [trade_returns, midprice_returns]:
        if returns_df is midprice_returns:
            returns_df_1s = midprice_returns_1s
        else:
            returns_df_1s = trade_returns_1s
            
        #Create 5-seccond and 15-second returns
        log_returns_5s = process_resample_data(returns_df, '5s', args.base_date)
        log_returns_15s = process_resample_data(returns_df, '15s', args.base_date)

        #Compute variances in minute bars on the second-returns
        ratios_1 = pd.DataFrame()
        ratios_1 = apply_ret_variances_aggregations(returns_df_1s)
        ratios_5 = pd.DataFrame()
        ratios_5 = apply_ret_variances_aggregations(log_returns_5s)
        ratios_15 = pd.DataFrame()
        ratios_15 = apply_ret_variances_aggregations(log_returns_15s)

        #Name the variance columns
        if (ratios_1 is not None and not ratios_1.empty) or (ratios_5 is not None and not ratios_5.empty) or (ratios_15 is not None and not ratios_15.empty):
            if ratios_1 is not None and not ratios_1.empty:
                ratios_1.rename(columns={"variance": "variance_1s"}, inplace=True)
            if ratios_5 is not None and not ratios_5.empty:
                ratios_5.rename(columns={"variance": "variance_5s"}, inplace=True)
            if ratios_15 is not None and not ratios_15.empty:
                ratios_15.rename(columns={"variance": "variance_15s"}, inplace=True)

            # Merge the two variances on the time index and find the two variance ratios
            if ratios_5 is not None and not ratios_5.empty and ratios_15 is not None and not ratios_15.empty:
                variance_ratio_df = pd.merge(ratios_5, ratios_15, left_index=True, right_index=True)
                
                #For variance ratio 1 the variances of 5 second and 15 second returns are divided
                variance_ratio_df['variance_ratio_s'] = np.abs((variance_ratio_df['variance_15s'] / (3 * variance_ratio_df['variance_5s'])) - 1)
                if ratios_1 is not None and not ratios_1.empty: 
                    variance_ratio_df = pd.merge(variance_ratio_df, ratios_1, left_index=True, right_index=True)
                    #For variance ratio 1 the variances of 1 second and 5 second returns are divided
                    variance_ratio_df['variance_ratio2_s'] = np.abs((variance_ratio_df['variance_5s'] / (5 * variance_ratio_df['variance_1s'])) - 1)
                    
                if returns_df is trade_returns:
                    aggregated_data["trade_ret_variance_ratio"] = reindex_to_full_time(variance_ratio_df['variance_ratio_s'],  args.base_date)
                    print(f"Variance ratio 1 was calculated for trade/price returns")
                    if ratios_1 is not None and not ratios_1.empty:
                        aggregated_data["trade_ret_variance_ratio2"] = reindex_to_full_time(variance_ratio_df['variance_ratio2_s'],  args.base_date)
                        print(f"Variance ratio 2 was calculated for trade/price returns")

                else:
                    aggregated_data["midprice_ret_variance_ratio"] = reindex_to_full_time(variance_ratio_df['variance_ratio_s'],  args.base_date)
                    print(f"Variance ratio 1 was calculated for midprice returns")
                    if ratios_1 is not None and not ratios_1.empty:
                        aggregated_data["midprice_ret_variance_ratio2"] = reindex_to_full_time(variance_ratio_df['variance_ratio2_s'],  args.base_date)
                        print(f"Variance ratio 2 was calculated for midprice returns")
                        
            else:
                print(f"Missing required columns for variance ratio calculation")
        else:
            print(f"Cammot compute variance ratios for since there is only one second-return for that day")
    end_process_vr_returns_time = time.time()

    #All variables have been saved to aggregated_data list
    
    #End calculation time
    main_end_time = time.time()

    write_start_time = time.time()
    
    # Function to merge dataframes on time index
    def merge_dataframes(df1, df2):
        return pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    
    #Split the saved datasets in aggregated data to datasets specific for trades, Buys_trades, Selld_trades, Retail trades ...
    categories = {
        "Trades": {"trades", "Herfindahl_trades", "Orderflow", "trade_returns", "trade_sign_stat", "trade_ret_variance_ratio", "trade_ret_variance_ratio2"},
        "Buys_trades": {"Buys_trades"},
        "Sells_trades": {"Sells_trades"},
        "Retail_trades": {"Retail_trades", "Orderflow_Retail"},
        "Buys_Retail_trades": {"Buys_Retail_trades"},
        "Sells_Retail_trades": {"Sells_Retail_trades"},
        "Oddlot_trades": {"Oddlot_trades", "Orderflow_Oddlot"},
        "Buys_Oddlot_trades": {"Buys_Oddlot_trades"},
        "Sells_Oddlot_trades": {"Sells_Oddlot_trades"},
        "Ask": {"Ask", "Herfindahl_Ask"},
        "Bid": {"Bid", "Herfindahl_Bid"},
        "Midpoint": {"Midpoint", "midprice_returns", "nbbo_sign_stat", "midprice_ret_variance_ratio", "midprice_ret_variance_ratio2"},
    }

    for category in categories:
        print(category)
        exec(f"merged_{category} = pd.DataFrame()")
        exec(f"merged_outside_trading_{category} = pd.DataFrame()")


    for name, df in aggregated_data.items():
        if df is not None and not df.isna().all().all():
            if name in categories["Trades"]:
                category = "Trades"
            elif name in categories["Buys_trades"]:
                category = "Buys_trades"
            elif name in categories["Sells_trades"]:
                category = "Sells_trades"
            elif name in categories["Retail_trades"]:
                category = "Retail_trades"
            elif name in categories["Buys_Retail_trades"]:
                category = "Buys_Retail_trades"
            elif name in categories["Sells_Retail_trades"]:
                category = "Sells_Retail_trades"
            elif name in categories["Oddlot_trades"]:
                category = "Oddlot_trades"
            elif name in categories["Buys_Oddlot_trades"]:
                category = "Buys_Oddlot_trades"
            elif name in categories["Sells_Oddlot_trades"]:
                category = "Sells_Oddlot_trades"
            elif name in categories["Ask"]:
                category = "Ask"
            elif name in categories["Bid"]:
                category = "Bid"
            elif name in categories["Midpoint"]:
                category = "Midpoint" 
            else:
                continue  # If the name doesn't match any category, skip it

        # Merge dataframes if the consolidated dataframe for the category is not empty, else assign df to it
            if eval(f"not merged_{category}.empty"):
                exec(f"merged_{category} = merge_dataframes(merged_{category}, df)")
            else:
                exec(f"merged_{category} = df")

    for name, df in aggregated_data_outside_trading.items():
        if df is not None and not df.isna().all().all():
            if name in categories["Trades"]:
                category = "Trades"
            elif name in categories["Buys_trades"]:
                category = "Buys_trades"
            elif name in categories["Sells_trades"]:
                category = "Sells_trades"
            elif name in categories["Retail_trades"]:
                category = "Retail_trades"
            elif name in categories["Buys_Retail_trades"]:
                category = "Buys_Retail_trades"
            elif name in categories["Sells_Retail_trades"]:
                category = "Sells_Retail_trades"
            elif name in categories["Oddlot_trades"]:
                category = "Oddlot_trades"
            elif name in categories["Buys_Oddlot_trades"]:
                category = "Buys_Oddlot_trades"
            elif name in categories["Sells_Oddlot_trades"]:
                category = "Sells_Oddlot_trades"
            elif name in categories["Ask"]:
                category = "Ask"
            elif name in categories["Bid"]:
                category = "Bid"
            elif name in categories["Midpoint"]:
                category = "Midpoint" 
            else:
                continue  # If the name doesn't match any category, skip it

        # Merge dataframes if the consolidated dataframe for the category is not empty, else assign df to it
            if eval(f"not merged_outside_trading_{category}.empty"):
                exec(f"merged_outside_trading_{category} = merge_dataframes(merged_outside_trading_{category}, df)")
            else:
                exec(f"merged_outside_trading_{category} = df")


    #Function for saving the variable datasets in a new HDF5 file
    def process_and_save_df(df, hdf5_variable_path, stock_name, day, month, year, time_range_name, category_name=None):
        if not df.empty:
            # Convert object columns to string
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str)
            
            # Convert datetime columns to formatted string and collect their names
            datetime_columns = []
            for col in df.columns:
                if df[col].dtype == "datetime64[ns]":
                    df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                    datetime_columns.append(col)
            
            # Save data to HDF5 file
            with pd.HDFStore(hdf5_variable_path, mode="a", complevel=9, complib="zlib") as store:
                hdf5_key = f"/{stock_name}/day{day}/{time_range_name}"
                if category_name:
                    hdf5_key += f"/{category_name}"
                store.append(hdf5_key, df, format="table", data_columns=True, index=False)
                print(f"Data successfully saved to HDF5 key: {hdf5_key}")
        else:
            message = f"{stock_name} has empty time bars for {day}/{month}/{year} and category: {category_name} {time_range_name}.\n"         
            try:
                with open(args.emp_analysis_path, "a") as f:
                    f.write(message)
                print(f"Message written to {args.emp_analysis_path}")
            except IOError as e:
                print(f"An error occurred while writing to the file: {e}")


    #Call saving function for groups daily_auction, inside_trading, outside_trading
    if auction_conditions_df is not None:
        process_and_save_df(auction_conditions_df, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "daily_auction")
    if daily_inside_df is not None:
        process_and_save_df(daily_inside_df, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "daily_trade_summary", "inside_trading")
    if daily_outside_df is not None:
        process_and_save_df(daily_outside_df, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "daily_trade_summary", "outside_trading")

    for category in categories:
        df = eval(f"merged_{category}")
        if df is not None:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'time'}, inplace=True)
            process_and_save_df(df, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "inside_trading", category)

        df_outside = eval(f"merged_outside_trading_{category}")
        if df_outside is not None:
            df_outside.reset_index(inplace=True)
            df_outside.rename(columns={'index': 'time'}, inplace=True)
            process_and_save_df(df_outside, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "outside_trading", category)

        write_end_time = time.time()

    #Write the time analysis to a text file
    if args.var_analysis_path is not None:
        with open(args.var_analysis_path, "a") as f:
            f.write(f"Stock: {args.stock_name}\n")
            f.write(f"Day: {args.day}\n")
            f.write(f"Only the calculation runtime: {main_end_time - main_start_time} seconds\n")
            f.write(f"Only the auction processing: {end_auction_time - start_auction_time} seconds\n")
            f.write(f"Only the trade processing: {end_process_trades_time - start_process_trades_time} seconds\n")
            f.write(f"OIB processing: {end_process_OIB_time - start_process_OIB_time} seconds\n")
            f.write(f"Herfindahl Index processing: {end_process_herfindahl_time- start_process_herfindahl_time} seconds\n")
            f.write(f"Only the quote processing: {end_process_quotes_time - start_process_quotes_time} seconds\n")
            f.write(f"Only the midpoint processing: {end_process_midpoint_time - start_process_midpoint_time} seconds\n")
            f.write(f"Only the return processing: {end_process_returns_time - start_process_returns_time} seconds\n")
            f.write(f"Only the variance ratios processing: {end_process_vr_returns_time - start_process_vr_returns_time} seconds\n")
            f.write(f"Write runtime: {write_end_time - write_start_time} seconds\n")

if __name__ == "__main__":
    print(f"Profiling path: {args.prof_analysis_path}")  # Debug statement to check profiling path
    if args.prof_analysis_path is not None:
        # Profile the main function
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()

        # Save profiling results
        try:
            with open(args.prof_analysis_path, "a") as f:
                f.write(f"\nStock: {args.stock_name}\n")
                ps = pstats.Stats(pr, stream=f)
                ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
            print(f"Profiling data written to {args.prof_analysis_path}")  # Confirm profiling data is written
        except IOError as e:
            print(f"An error occurred while writing the profiling data: {e}")
    else:
        main()
