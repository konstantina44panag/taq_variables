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
from preparation import prepare_datasets, NoTradesException, NoNbbosException
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

        trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs = result

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
        special_conditions_df = pl_df.filter(pl.col('cond').str.contains('Q|O|5|M|6|9'))     #these codes correspond to opening and closing prices 
        return special_conditions_df.select(['time', 'price', 'vol', 'EX', 'cond']).to_pandas()

    #Calculate the variance
    def calculate_minute_volatility(returns):
        n = len(returns)
        if n <= 1:
            return np.nan
        mean_return = returns.mean()
        return ((returns - mean_return) ** 2).sum() / (len(returns) - 1)

    #Calculate the autocorrelation
    def calculate_autocorrelation(returns):
        n = len(returns)
        if n <= 1:
            return np.nan
        mean = returns.mean()
        variance = ((returns - mean) ** 2).sum() / (n - 1)
        if np.isclose(variance, 0):
            return np.nan
        covariance = ((returns - mean) * (returns.shift(1) - mean)).sum() / (n - 1)
        return covariance / variance

    #Calculate the orderflow by Chordia, Hu, Subrahmanyam and Tong, MS 2019
    def calculate_voib_shr(df1, df2):
        if df1 is None or df1.empty or df1.isna().all().all() or df2 is None or df2.empty or df2.isna().all().all():
            return None
        
        if 'time' in df1.columns:
            df1.set_index('time', inplace=True)
        if 'time' in df2.columns:
            df2.set_index('time', inplace=True)

        df1_filtered = df1.between_time("09:30", "16:00").copy()
        df2_filtered = df2.between_time("09:30", "16:00").copy()
        
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
        pl.col('OIB_SHR').map_elements(calculate_minute_volatility, return_dtype=pl.Float64).alias('VOIB_SHR'),
        pl.col('OIB_SHR').map_elements(calculate_autocorrelation, return_dtype=pl.Float64).alias('OIB_SHR_autocorr'),
        ])
        return resampled_df.to_pandas().set_index('time')
            
    #Calculate the variance and autocorrelation of returns      
    def apply_return_aggregations(pl_df, column='returns', df_name=''):
        if pl_df is None or pl_df.shape[0] == 0:
            return None
        if pl_df.shape[0] == 1:
            return None
        volatility_col_name = f'{df_name}_volatility'
        autocorr_col_name = f'{df_name}_autocorr'
        resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left').agg([
            pl.col(column).map_elements(calculate_minute_volatility, return_dtype=pl.Float64).alias(volatility_col_name),
            pl.col(column).map_elements(calculate_autocorrelation, return_dtype=pl.Float64).alias(autocorr_col_name),
        ])
        return resampled_df.to_pandas().set_index('time')
        
    #Calculate the variance and autocorrelation of returns outside trading hours          
    def apply_return_aggregations_outside_trading(pl_df, column='returns',df_name=''):
        if pl_df is None or pl_df.shape[0] == 0:
            return None
        if pl_df.shape[0] == 1:
            return None
        volatility_col_name = f'{df_name}_volatility'
        autocorr_col_name = f'{df_name}_autocorr'
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
        pl.col(column).map_elements(calculate_minute_volatility, return_dtype=pl.Float64).alias('variance')])
           
        return resampled_df.to_pandas().set_index('time')
    
    #Fill the minutes in the time column after the time bars are calculated for different dataframes and variables, so that entire time bar data can be merged in one dataframe
    def reindex_to_full_time(df, base_date, outside_trading=False):
        if df is None or df.empty or df.isna().all().all():
            return None
        
        if outside_trading:
            morning_index = pd.date_range(start=f"{base_date} 03:30", end=f"{base_date} 09:30", freq="30min")
            evening_index = pd.date_range(start=f"{base_date} 16:00", end=f"{base_date} 20:00", freq="30min")
            full_time_index = morning_index.union(evening_index)
        else:
            full_time_index = pd.date_range(start=f"{base_date} 09:30", end=f"{base_date} 16:00", freq="1min")
        df_reindexed = df.reindex(full_time_index)
        return df_reindexed

    # Aggregated Functions for Trades
    #Variables for the trades dataframes, most of them are formed at the same time 
    def apply_aggregations(df, df_name):
        #Check for empty dataframe or for dimension size 1 when there is no reason for creating minute bars
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df
        df_filtered = df.copy()
           
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
                pl.col('count').max().alias(f'{df_name}_max_events_per_sec')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()
                
            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))
                
            #Unite the different variables/aggregation methods, to be computed at the same time
            aggregations = [
                pl.col('price').last().alias(f'{df_name}_last_price'),
                pl.col('vol').last().alias(f'{df_name}_last_vol'),
                pl.col('time').last().alias(f'{df_name}_last_time'),
                pl.col('price').mean().alias(f'{df_name}_avg_price'),
                pl.col('vol').mean().alias(f'{df_name}_avg_vol'),
                pl.col('vol').sum().alias(f'{df_name}_tot_vol'),
                calculate_vwap_pl().alias(f'{df_name}_vwap'),
                calculate_twap_pl().alias(f'{df_name}_twap'),
                pl.count('price').alias(f'{df_name}_num_events')
            ]

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
    def apply_aggregations_outside_trading(df, df_name, base_date):
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df

        df_filtered = df.copy()
        
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
                pl.col('count').max().alias(f'{df_name}_max_events_per_sec')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()

            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))
            
            aggregations = [
                pl.col('price').last().alias(f'{df_name}_last_price'),
                pl.col('vol').last().alias(f'{df_name}_last_vol'),
                pl.col('time').last().alias(f'{df_name}_last_time'),
                pl.col('price').mean().alias(f'{df_name}_avg_price'),
                pl.col('vol').mean().alias(f'{df_name}_avg_vol'),
                pl.col('vol').sum().alias(f'{df_name}_tot_vol'),
                calculate_vwap_pl().alias(f'{df_name}_vwap'),
                calculate_twap_pl().alias(f'{df_name}_twap'),
                pl.count('price').alias(f'{df_name}_num_events')
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='30m', closed='left', label='left').agg(aggregations)

            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    #Aggregated Functions for Quotes
    def apply_quote_aggregations(df, df_name):
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df

        df_filtered = df.copy()

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
                pl.col('count').max().alias(f'{df_name}_max_events_per_sec')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()

            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))
                
            #Find the trading halts, or news event indicator
            def encode_conditions_expr():
                return pl.col('qu_cond').map_elements(lambda x: ''.join([c for c in x if c in 'DIJKLMNOPQRSTVYZ124']), return_dtype=pl.Utf8)  #codes for trading halts and reopenings
            
            aggregations = [
                pl.col('price').last().alias(f'{df_name}_last_price'),
                pl.col('vol').last().alias(f'{df_name}_last_vol'),
                pl.col('time').last().alias(f'{df_name}_last_time'),
                pl.col('price').mean().alias(f'{df_name}_avg_price'),
                pl.col('vol').mean().alias(f'{df_name}_avg_vol'),
                pl.col('vol').sum().alias(f'{df_name}_tot_vol'),
                calculate_vwap_pl().alias(f'{df_name}_vwap'),
                calculate_twap_pl().alias(f'{df_name}_twap'),
                pl.count('price').alias(f'{df_name}_num_events'),
                encode_conditions_expr().alias(f'{df_name}_halt_indic')
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def apply_quote_aggregations_outside_trading(df, df_name, base_date):
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df
        
        df_filtered = df.copy()

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
                pl.col('count').max().alias(f'{df_name}_max_events_per_sec')
            ])

            #Compute the VWAP
            def calculate_vwap_pl():
                return (pl.col('price') * pl.col('vol')).sum() / pl.col('vol').sum()

            #Compute the TWAP
            def calculate_twap_pl():
                return (pl.sum('weighted_price') / pl.sum('durations'))

            #Find the trading halts, or news event indicator
            def encode_conditions_expr():
                return pl.col('qu_cond').map_elements(lambda x: ''.join([c for c in x if c in 'DIJKLMNOPQRSTVYZ124']), return_dtype=pl.Utf8)
            
            aggregations = [
                pl.col('price').last().alias(f'{df_name}_last_price'),
                pl.col('vol').last().alias(f'{df_name}_last_vol'),
                pl.col('time').last().alias(f'{df_name}_last_time'),
                pl.col('price').mean().alias(f'{df_name}_avg_price'),
                pl.col('vol').mean().alias(f'{df_name}_avg_vol'),
                pl.col('vol').sum().alias(f'{df_name}_tot_vol'),
                calculate_vwap_pl().alias(f'{df_name}_vwap'),
                calculate_twap_pl().alias(f'{df_name}_twap'),
                pl.count('price').alias(f'{df_name}_num_events'),
                encode_conditions_expr().alias(f'{df_name}_halt_indic')
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='30m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    #Functions for Midprice

    def apply_midpoint_aggregations(df):
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df
        

        df_filtered = df.copy()

        pl_df = pl.from_pandas(df_filtered.reset_index())

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='1m', label='left').agg([
                pl.col('count').max().alias(f'midprice_max_events_per_sec')
            ])

            #Count the number of events
            aggregations = [
                pl.count('price').alias(f'midprice_num_events'),
            ]

            resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg(aggregations)
            resampled_df = resampled_df.join(max_trades_per_sec, on='time', how='inner')
            return resampled_df.to_pandas().set_index('time')
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return
        


    def apply_midpoint_aggregations_outside_trading(df, base_date):
        if df is None or df.empty or df.isna().all().all():
            return None
        if len(df) == 1:
            return df
       
        df_filtered = df.copy()

        pl_df = pl.from_pandas(df_filtered.reset_index())

        try:
            #Group the dataframe to 1-second intervals and count the events inside
            seconds_df = pl_df.group_by_dynamic('time', every='1s', label='left').agg([
                pl.count('price').alias('count')
            ])

            #Compute the variable: maximum events in the seconds per minute
            max_trades_per_sec = seconds_df.group_by_dynamic('time', every='30m', label='left').agg([
                pl.col('count').max().alias(f'midprice_max_events_per_sec')
            ])

            #Count the number of events
            aggregations = [
                pl.count('price').alias(f'midprice_num_events'),
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
            df_filtered = df[(df.index < start_time_morning) | (df.index > end_time_afternoon)].copy()
        else:
            df_filtered = df.between_time("09:30", "16:00").copy()
            
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


    # Processing Trades

    #Apply the function for open /close prices
    start_auction_time = time.time()
    auction_conditions_df = auction_conditions(trades)
    end_auction_time = time.time()

    #For every trade dataframe, apply the function apply_aggregations from above
    start_process_trades_time = time.time()

    trade_dataframes_to_process = {
        "trades": trades,
        "Buys_trades": Buys_trades,
        "Sells_trades": Sells_trades,
        "Retail_trades": Retail_trades,
        "Oddlot_trades": Oddlot_trades,
    }

    for name, df in trade_dataframes_to_process.items():
        if df is None or df.empty or df.isna().all().all():
            continue
         
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
      
        df_filtered = df.between_time("09:30", "16:00").copy()
        start_time_morning = f"{args.base_date} 09:30"
        end_time_afternoon = f"{args.base_date} 16:00"
        df_filtered_outside = df[(df.index < start_time_morning) | (df.index > end_time_afternoon)].copy()

        if not df_filtered.empty:
            try:
                print(f"Processing {name} DataFrame")
                agg_df = apply_aggregations(df, name)
                aggregated_data[name] = reindex_to_full_time(agg_df, args.base_date)                
            except KeyError as e:
                print(f"Error processing {name}: {e}")
                continue

        if not df_filtered_outside.empty:
            try:
                print(f"Processing {name} DataFrame outside trading hours")
                agg_df_outside_trading = apply_aggregations_outside_trading(df, name, args.base_date)
                print(agg_df_outside_trading)
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
    oib_shr_s = calculate_voib_shr(Buys_trades, Sells_trades)
    if oib_shr_s is not None:
        oib_shr_df = oib_shr_s.to_frame(name='OIB_SHR').reset_index()
        oib_shr_df.columns = ['time', 'OIB_SHR']
    else:
        oib_shr_df = pd.DataFrame(columns=['time', 'OIB_SHR'])

    #Orderflow Statistics (based on the traded volume)
    aggregated_data["OIB_SHR"] = reindex_to_full_time(apply_voib_shr_aggregations(oib_shr_df), args.base_date)
    
    end_process_OIB_time = time.time()

    #Herfindahl Index is performed seperately
    start_process_herfindahl_time = time.time()
    print(f"Processing Herfindahl Index")

    def calculate_hindex(df, name):
        if df is None or df.empty or df.isna().all().all():
            return None
        if 'time' in df.columns:
            df.set_index('time', inplace=True)

        df_filtered = df.between_time('09:30', '16:00').copy()
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
            (minutely_data['sum_of_squared_values'] / minutely_data['sum_of_values_squared']).alias('proportion')
        ])
        
        proportion_column_name = f'proportion_{name}'
        minutely_data = minutely_data.select([
            'time', 'proportion'
        ]).rename({
            'proportion': proportion_column_name
        })
        
        aggregated_data[f"hindex_{name}"] = reindex_to_full_time(minutely_data.to_pandas().set_index('time'), args.base_date)


    #Apply the Herfindahl Index for these dataframes
    for df, name in zip([trades, Buys_trades, Sells_trades, Retail_trades, Oddlot_trades, Ask, Bid], 
                        ["trades", "Buys_trades", "Sells_trades", "Retail_trades", "Oddlot_trades", "Ask", "Bid"]):
        calculate_hindex(df, name)
       
    end_process_herfindahl_time = time.time()

    #Processing Midpoint, apply the function apply_midpoint_aggregations from above
    start_process_midpoint_time = time.time()

    if not Midpoint.empty:

        if 'time' in df.columns:
            df.set_index('time', inplace=True)
      
        df_filtered = df.between_time("09:30", "16:00").copy()
        start_time_morning = f"{args.base_date} 09:30"
        end_time_afternoon = f"{args.base_date} 16:00"
        df_filtered_outside = df[(df.index < start_time_morning) | (df.index > end_time_afternoon)].copy()
        
        if not df_filtered.empty:
            try:
                print(f"Processing Midpoint DataFrame")
                midpoint_agg_df = apply_midpoint_aggregations(Midpoint)
                aggregated_data["Midpoint"] = reindex_to_full_time(midpoint_agg_df, args.base_date)
            except KeyError as e:
                print(f"Error processing Midpoint: {e}")

        if not df_filtered_outside.empty:
            try:
                print(f"Processing Midpoint DataFrame outside trading hours")
                midpoint_agg_df_outside_trading = apply_midpoint_aggregations_outside_trading(Midpoint, args.base_date)
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
      
        df_filtered = df.between_time("09:30", "16:00").copy()
        start_time_morning = f"{args.base_date} 09:30"
        end_time_afternoon = f"{args.base_date} 16:00"
        df_filtered_outside = df[(df.index < start_time_morning) | (df.index > end_time_afternoon)].copy()

        if not df_filtered.empty:
            try:
                print(f"Processing {name} DataFrame")
                agg_df = apply_quote_aggregations(df, name)
                aggregated_data[name] = reindex_to_full_time(agg_df,  args.base_date)          
            except KeyError as e:
                print(f"Error processing {name}: {e}")
                continue

        if not df_filtered_outside.empty:
            try:
                print(f"Processing {name} DataFrame outside trading hours")
                agg_df_outside_trading = apply_quote_aggregations_outside_trading(df, name, args.base_date)
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
    aggregated_data["trade_returns"] = reindex_to_full_time(apply_return_aggregations(trade_returns_1s, column='returns', df_name='trade_ret'), args.base_date)
    aggregated_data["midprice_returns"] = reindex_to_full_time(apply_return_aggregations(midprice_returns_1s, column='returns', df_name='midprice_ret'),  args.base_date)
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
                variance_ratio_df['variance_ratio'] = np.abs((variance_ratio_df['variance_15s'] / (3 * variance_ratio_df['variance_5s'])) - 1)
                if ratios_1 is not None and not ratios_1.empty: 
                    variance_ratio_df = pd.merge(variance_ratio_df, ratios_1, left_index=True, right_index=True)
                    
                    #For variance ratio 1 the variances of 1 second and 5 second returns are divided
                    variance_ratio_df['variance_ratio2'] = np.abs((variance_ratio_df['variance_5s'] / (5 * variance_ratio_df['variance_1s'])) - 1)
                if returns_df is trade_returns:
                    variance_ratio_df.rename(columns={'variance_ratio': 'trade_ret_variance_ratio'}, inplace=True)
                    aggregated_data["trade_ret_variance_ratio"] = reindex_to_full_time(variance_ratio_df['trade_ret_variance_ratio'],  args.base_date)
                    print(f"Variance ratio 1 was calculated for trade/price returns")
                    if ratios_1 is not None and not ratios_1.empty:
                        variance_ratio_df.rename(columns={'variance_ratio2': 'trade_ret_variance_ratio2'}, inplace=True)
                        aggregated_data["trade_ret_variance_ratio2"] = reindex_to_full_time(variance_ratio_df['trade_ret_variance_ratio2'],  args.base_date)
                        print(f"Variance ratio 2 was calculated for trade/price returns")

                else:
                    variance_ratio_df.rename(columns={'variance_ratio': 'midprice_ret_variance_ratio'}, inplace=True)
                    aggregated_data["midprice_ret_variance_ratio"] = reindex_to_full_time(variance_ratio_df['midprice_ret_variance_ratio'],  args.base_date)
                    print(f"Variance ratio 1 was calculated for midprice returns")
                    if ratios_1 is not None and not ratios_1.empty:
                        variance_ratio_df.rename(columns={'variance_ratio2': 'midprice_ret_variance_ratio2'}, inplace=True)
                        aggregated_data["midprice_ret_variance_ratio2"] = reindex_to_full_time(variance_ratio_df['midprice_ret_variance_ratio2'],  args.base_date)
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
    
    consolidated_df = pd.DataFrame()
    
    #Create a single dataframe 'consolidated_df' for all names and dataframes in the aggregated_data list
    for name, df in aggregated_data.items():
        if df is not None and not df.isna().all().all():
            consolidated_df = merge_dataframes(consolidated_df, df) if not consolidated_df.empty else df

    #Create a single dataframe 'consolidated_df_outside_trading' for all names and dataframes in the aggregated_data list
    consolidated_df_outside_trading = pd.DataFrame()
    for name, df in aggregated_data_outside_trading.items():
        if df is not None and not df.isna().all().all():
            consolidated_df_outside_trading = merge_dataframes(consolidated_df_outside_trading, df) if not consolidated_df_outside_trading.empty else df

    #Reset the time index to be the time column of the variable dataset
    consolidated_df.reset_index(inplace=True)
    consolidated_df.rename(columns={'index': 'time'}, inplace=True)

    consolidated_df_outside_trading.reset_index(inplace=True)
    consolidated_df_outside_trading.rename(columns={'index': 'time'}, inplace=True)

    # Debug: Check individual DataFrames in aggregated_data_outside_trading
    #for name, df in aggregated_data_outside_trading.items():
    #    print(f"DataFrame: {name}")
    #    print(df.info())

    #Function for saving the variable datasets in a new HDF5 file
    def process_and_save_df(df, hdf5_variable_path, stock_name, day, month, year, time_range_name):
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
            print(f"Saving data to HDF5 file: {hdf5_variable_path}")
            with pd.HDFStore(hdf5_variable_path, mode="a", complevel=9, complib="zlib") as store:
                hdf5_key = f"/{stock_name}/day{day}/{time_range_name}"
                store.append(hdf5_key, df, format="table", data_columns=True, index=False)
                print(f"Data successfully saved to HDF5 key: {hdf5_key}")
        else:
            print("No DataFrames to merge. Skipping HDF5 save step.")
            message = f"{stock_name} has empty time bars for {day}/{month}/{year}."
            
            try:
                with open(args.emp_analysis_path, "w") as f:
                    f.write(message)
                print(f"Message written to {args.emp_analysis_path}")
            except IOError as e:
                print(f"An error occurred while writing to the file: {e}")

    #Save the variables inside trading hours in the key time_bars for that stock and that day
    if consolidated_df is not None and not consolidated_df.empty:
        process_and_save_df(consolidated_df, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "time_bars")
    else:
        print("Consolidated DataFrame is empty or None. Skipping save.")

    #Save the variables outside trading hours in the key outside_trading_time_bars for that stock and that day
    if consolidated_df_outside_trading is not None and not consolidated_df_outside_trading.empty:
        process_and_save_df(consolidated_df_outside_trading, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "outside_trading_time_bars")
    else:
        print("Consolidated DataFrame outside trading is empty or None. Skipping save.")

    if auction_conditions_df is not None and not auction_conditions_df.empty:
        process_and_save_df(auction_conditions_df, args.hdf5_variable_path, args.stock_name, args.day, args.month, args.year, "daily_auction")
    else:
        print("Daily auction, open and close prices not found")

    write_end_time = time.time()

    #Write the time analysis to a text file
    if args.var_analysis_path:
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
    if args.prof_analysis_path:
        # Profile the main function
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()

        # Save profiling results
        with open(args.prof_analysis_path, "a") as f:
            f.write(f"\nStock: {args.stock_name}\n")
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
    else:
        main()
