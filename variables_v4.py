#!/usr/bin/env python3.11
"""This script computes important variables but first passes arguments to preparation.py."""

import pandas as pd
import numpy as np
import argparse
import cProfile
import pstats
import argparse
import time
import polars as pl
from preparation import prepare_datasets
from aggregation_functions import auction_conditions, reindex_to_full_time, flatten_dict, calculate_oib_metrics, apply_oib_aggregations, apply_return_aggregations,apply_ret_variances_aggregations, apply_aggregations, apply_quote_aggregations, apply_midpoint_aggregations, process_resample_data, process_daily, calculate_Herfindahl

# Parse arguments
parser = argparse.ArgumentParser(
    description="Prepare datasets for trade sign analysis and variable estimation."
)
parser.add_argument(
    "hdf5_file_path", type=str, help="The path to the original HDF5 file."
)
parser.add_argument("base_date", type=str, help="Base date for the analysis.")
parser.add_argument("stock_name", type=str, help="Stock symbol.")
parser.add_argument("s", type=str, help="Stock suffix.")
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
            args.s,
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

        if result is not None:
            trades, Buys_trades, Sells_trades, Ask, Bid, Retail_trades, Oddlot_trades, Buys_Oddlot_trades, Sells_Oddlot_trades, Buys_Retail_trades, Sells_Retail_trades, Midpoint, trade_returns, midprice_returns, trade_signs, nbbo_signs = result
        else:
            print(f"No nbbos or trades for {args.stock_name} on {args.base_date}.")
            return
    except Exception as e:
        print(f"An error occurred while preparing datasets: {e}")
        raise
    
    # Start timing the main calculations
    main_start_time = time.time()
   
    #Process daily bars
    exchanges_set = list('ABCDHIJKLMNPSTQUVWXYZ1')    #DAILY TAQ MANUALS: 2024, 2022, 2020, 2016, 2013,  2006
    conditions_set = [''] + ['nan'] + list('@ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')   #DAILY TAQ MANUALS: 2024, 2022, 2020, 2016, 2013,  2006
    if 'time' in trades.columns:
            trades.set_index('time', inplace=True)
    df_filtered_inside  = trades.between_time("09:30", "15:59:59").copy()
    start_time_morning = f"{args.base_date} 09:30"
    end_time_afternoon = f"{args.base_date} 16:00"
    df_filtered_outside = trades[(trades.index < start_time_morning) | (trades.index >= end_time_afternoon)].copy()
    trades.reset_index(inplace=True)
    df_filtered_inside =  pl.from_pandas(df_filtered_inside)
    df_filtered_outside = pl.from_pandas(df_filtered_outside)
    daily_inside_df_cond, daily_outside_df_cond = process_daily(df_filtered_inside, df_filtered_outside, conditions_set, 'cond', is_cond=True)
    daily_inside_df_ex, daily_outside_df_ex = process_daily(df_filtered_inside, df_filtered_outside, exchanges_set,'EX', is_cond=False)
    flat_daily_inside_cond = flatten_dict(daily_inside_df_cond)
    flat_daily_outside_cond = flatten_dict(daily_outside_df_cond)
    flat_daily_inside_ex = flatten_dict(daily_inside_df_ex)
    flat_daily_outside_ex = flatten_dict(daily_outside_df_ex)
    daily_inside_cond = pd.DataFrame([flat_daily_inside_cond])
    daily_outside_cond = pd.DataFrame([flat_daily_outside_cond])
    daily_inside_ex = pd.DataFrame([flat_daily_inside_ex])
    daily_outside_ex = pd.DataFrame([flat_daily_inside_ex])
    daily_inside_df = pd.concat([daily_inside_cond, daily_inside_ex], axis=1)
    daily_outside_df = pd.concat([daily_outside_cond, daily_outside_ex], axis=1)

    #Cleaning step T2, clear trade conditions after the computation of daily bars
    
    pl_trades = pl.from_pandas(trades)
    pl_trades = pl_trades.filter(~pl_trades['cond'].str.contains('B|G|J|K|L|W|Z'))
    
    #Check for empty dataframe after the cleaning step
    if pl_trades.height == 0:
        print(f"No trades after cleaning techniques for {args.stock_name}")
        return
    
    #Processing open /close auction prices
    start_auction_time = time.time()
    auction_conditions_df, trades = auction_conditions(pl_trades)
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
                agg_df = apply_aggregations(df_filtered, name)
                aggregated_data[name] = reindex_to_full_time(agg_df, args.base_date)                
            except KeyError as e:
                print(f"Error processing {name}: {e}")
                raise

        if not df_filtered_outside.empty:
            try:
                agg_df_outside_trading = apply_aggregations(df_filtered_outside, name, outside_trading=True)
                aggregated_data_outside_trading[name] = reindex_to_full_time(agg_df_outside_trading, args.base_date, outside_trading=True)
            except KeyError as e:
                print(f"Error processing {name}: {e} outside trading hours")
                raise
    end_process_trades_time = time.time()

    
    #The orderflow calculation is performed seperately
    start_process_OIB_time = time.time()
    # orderflow estimation from Buys and Sells
    trade_pairs = [
        (Buys_trades, Sells_trades, "Orderflow_Trades"),
        (Buys_Oddlot_trades, Sells_Oddlot_trades, "Orderflow_Oddlot"),
        (Buys_Retail_trades, Sells_Retail_trades, "Orderflow_Retail"),
        (Bid, Ask, "Orderflow_Quotes")
    ]

    for df1, df2, key in trade_pairs:
        if 'time' in df1.columns:
            df1.set_index('time', inplace=True)
        if 'time' in df2.columns:
            df2.set_index('time', inplace=True)
        df1_filtered = df1.between_time("09:30", "15:59:59").copy()
        df2_filtered = df2.between_time("09:30", "15:59:59").copy()
        oib_df = calculate_oib_metrics(df1_filtered, df2_filtered, args.base_date)
        oib_statistics = apply_oib_aggregations(oib_df)
        aggregated_data[key] = reindex_to_full_time(oib_statistics, args.base_date)

    end_process_OIB_time = time.time()

    #Herfindahl Index is performed seperately
    start_process_herfindahl_time = time.time()
    #Apply the Herfindahl Index for these dataframes
    for df, name in zip([trades, Ask, Bid], 
                        ["trades", "Ask", "Bid"]):
        herfindahl = calculate_Herfindahl(df)
        aggregated_data[f"Herfindahl_{name}"] = reindex_to_full_time(herfindahl, args.base_date)

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
                midpoint_agg_df = apply_midpoint_aggregations(df_filtered)
                aggregated_data["Midpoint"] = reindex_to_full_time(midpoint_agg_df, args.base_date)
            except KeyError as e:
                print(f"Error processing Midpoint: {e}")
                raise

        if not df_filtered_outside.empty:
            try:
                midpoint_agg_df_outside_trading = apply_midpoint_aggregations(df_filtered_outside, outside_trading=True)
                aggregated_data_outside_trading["Midpoint"] = reindex_to_full_time(midpoint_agg_df_outside_trading, args.base_date, outside_trading=True)
            except KeyError as e:
                print(f"Error processing Midpoint outside trading hours: {e}")
                raise
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
                agg_df = apply_quote_aggregations(df_filtered, name)
                aggregated_data[name] = reindex_to_full_time(agg_df,  args.base_date)          
            except KeyError as e:
                print(f"Error processing {name}: {e}")
                raise

        if not df_filtered_outside.empty:
            try:
                agg_df_outside_trading = apply_quote_aggregations(df_filtered_outside, name, outside_trading=True)
                aggregated_data_outside_trading[name] = reindex_to_full_time(agg_df_outside_trading,  args.base_date, outside_trading=True)
            except KeyError as e:
                print(f"Error processing {name}: {e} outside trading hours")
                raise
    end_process_quotes_time = time.time()
    

    #Processing Returns
    start_process_returns_time = time.time()

    #Create one second returns for the trade_returns (from trade prices),  for the midprice_returns (from quote midpoint), for the trade_signs and the nbbo_signs for inside and outside trading hours
    trade_returns_1s = process_resample_data(trade_returns, '1s', args.base_date)    
    midprice_returns_1s = process_resample_data(midprice_returns, '1s', args.base_date)
    trade_returns_1m_outside_trading = process_resample_data(trade_returns, '1m', args.base_date, outside_trading=True)
    midprice_returns_1m_outside_trading = process_resample_data(midprice_returns, '1m', args.base_date, outside_trading=True)

    nbbo_signs_1s = process_resample_data(nbbo_signs, '1s', args.base_date)
    trade_signs_1s = process_resample_data(trade_signs, '1s', args.base_date)
    nbbo_signs_1m_outside_trading = process_resample_data(nbbo_signs, '1m', args.base_date, outside_trading=True)
    trade_signs_1m_outside_trading = process_resample_data(trade_signs, '1m', args.base_date, outside_trading=True)

    #Create one minute timebars with statistics of the one second returns, such as variance and autocorrelation, for inside and outside trading hours
    aggregated_data["trade_returns"] = reindex_to_full_time(apply_return_aggregations(trade_returns_1s, column='returns'), args.base_date)
    aggregated_data["trade_signs"] = reindex_to_full_time(apply_return_aggregations(trade_signs_1s, column='returns',sign=True),  args.base_date)

    aggregated_data["midprice_returns"] = reindex_to_full_time(apply_return_aggregations(midprice_returns_1s, column='returns'),  args.base_date)
    aggregated_data["nbbo_signs"] = reindex_to_full_time(apply_return_aggregations(nbbo_signs_1s, column='returns', sign=True),  args.base_date)
    
    aggregated_data_outside_trading["trade_returns"] = reindex_to_full_time(apply_return_aggregations(trade_returns_1m_outside_trading, column='returns', outside_trading=True),  args.base_date, outside_trading=True)
    aggregated_data_outside_trading["trade_signs"] = reindex_to_full_time(apply_return_aggregations(trade_signs_1m_outside_trading, column='returns', sign=True, outside_trading=True),  args.base_date, outside_trading=True)

    aggregated_data_outside_trading["midprice_returns"] = reindex_to_full_time(apply_return_aggregations(midprice_returns_1m_outside_trading, column='returns', outside_trading=True),  args.base_date, outside_trading=True)
    aggregated_data_outside_trading["nbbo_signs"] = reindex_to_full_time(apply_return_aggregations(nbbo_signs_1m_outside_trading, column='returns', sign=True, outside_trading=True),  args.base_date, outside_trading=True)
    
    end_process_returns_time = time.time()

    #Variance Ratios
    start_process_vr_time = time.time()    
    for returns_df, name, name2 in zip([trade_returns, midprice_returns], ['trade_vr', 'midprice_vr'], ['trade_vr2', 'midprice_vr2']):
        if name == 'midprice_vr':
            returns_df_1s = midprice_returns_1s
        else:
            returns_df_1s = trade_returns_1s
            
        #Create 5-seccond and 15-second returns
        log_returns_5s = process_resample_data(returns_df, '5s', args.base_date)
        log_returns_15s = process_resample_data(returns_df, '15s', args.base_date)

        #Compute variances in minute bars on the second-returns
        var_1 = pd.DataFrame()
        var_1 = apply_ret_variances_aggregations(returns_df_1s,'variance_1s')
        var_5 = pd.DataFrame()
        var_5 = apply_ret_variances_aggregations(log_returns_5s,'variance_5s')
        var_15 = pd.DataFrame()
        var_15 = apply_ret_variances_aggregations(log_returns_15s, 'variance_15s')
        variance_ratio_df = None
        variance_ratio_df2 = None
        # Merge the two variances on the time index and find the two variance ratios
        if var_5 is not None and not var_5.empty and var_15 is not None and not var_15.empty:
            variance_ratio_df = pd.merge(var_5, var_15, left_index=True, right_index=True)
            variance_ratio_df['vratio_s'] = np.abs((variance_ratio_df['variance_15s'] / (3 * variance_ratio_df['variance_5s'])) - 1)

        if var_1 is not None and not var_1.empty and var_5 is not None and not var_5.empty: 
                variance_ratio_df2 = pd.merge(var_1, var_5, left_index=True, right_index=True)
                variance_ratio_df2['vratio2_s'] = np.abs((variance_ratio_df2['variance_5s'] / (5 * variance_ratio_df2['variance_1s'])) - 1)

        if variance_ratio_df is not None and not variance_ratio_df.empty:
            aggregated_data[name] = reindex_to_full_time(variance_ratio_df['vratio_s'],  args.base_date)
        if variance_ratio_df2 is not None and not variance_ratio_df2.empty:
            aggregated_data[name2] = reindex_to_full_time(variance_ratio_df2['vratio2_s'],  args.base_date)      
    end_process_vr_time = time.time()  
    #All variables have been saved to lists: aggregated_data (for inside trading) and aggregated_data_outside_trading.
    
    #End calculation time
    main_end_time = time.time()

    write_start_time = time.time()
    #Split the saved datasets in aggregated data to datasets specific for trades, Buys_trades, Selld_trades, Retail trades ...
    categories = {
        "Trades": {"trades", "Herfindahl_trades", "Orderflow_Trades", "trade_returns", "trade_signs", "trade_vr", "trade_vr2"},
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
        "Midpoint": {"Midpoint", "Orderflow_Quotes", "midprice_returns", "nbbo_signs", "midprice_vr", "midprice_vr2"},
    }

    def merge_dataframes(df1, df2):
        return pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    
    #Merge data of the same category
    merged_data = {category: pd.DataFrame() for category in categories}
    merged_outside_trading_data = {category: pd.DataFrame() for category in categories}   
    for category, names in categories.items():
        for name, df in aggregated_data.items():
            if name in names and df is not None and not df.isna().all().all():
                if not merged_data[category].empty:
                    merged_data[category] = merge_dataframes(merged_data[category], df)
                else:
                    merged_data[category] = df
        for name, df in aggregated_data_outside_trading.items():
            if name in names and df is not None and not df.isna().all().all():
                if not merged_outside_trading_data[category].empty:
                    merged_outside_trading_data[category] = merge_dataframes(merged_outside_trading_data[category], df)
                else:
                    merged_outside_trading_data[category] = df
    

    #Function for saving the variable datasets in a new HDF5 file
    def process_and_save_df(df, hdf5_variable_path, stock_name, s, day, month, year, time_range_name, category_name=None):
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
                hdf5_key = f"/{stock_name}/suffix_{s}/day{day}/{time_range_name}"
                if category_name:
                    hdf5_key += f"/{category_name}"
                store.append(hdf5_key, df, format="table", data_columns=True, index=False)
        else:
            message = f"{stock_name}, suffix {s}, has empty time bars for {day}/{month}/{year} and category: {category_name} {time_range_name}.\n"         
            try:
                with open(args.emp_analysis_path, "a") as f:
                    f.write(message)
            except IOError as e:
                print(f"An error occurred while writing to the file: {e}")
                raise


    #Call saving function for groups daily_auction, inside_trading, outside_trading
    if auction_conditions_df is not None:
        process_and_save_df(auction_conditions_df, args.hdf5_variable_path, args.stock_name, args.s, args.day, args.month, args.year, "daily_auction_summary")
    if daily_inside_df is not None:
        process_and_save_df(daily_inside_df, args.hdf5_variable_path, args.stock_name, args.s, args.day, args.month, args.year, "daily_trade_summary", "inside_trading")
    if daily_outside_df is not None:
        process_and_save_df(daily_outside_df, args.hdf5_variable_path, args.stock_name, args.s, args.day, args.month, args.year, "daily_trade_summary", "outside_trading")

    for category, df in merged_data.items():
        if df is not None:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'time'}, inplace=True)
            process_and_save_df(df, args.hdf5_variable_path, args.stock_name, args.s, args.day, args.month, args.year, "inside_trading", category)

    for category, df in merged_outside_trading_data.items():
        if df is not None:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'time'}, inplace=True)
            process_and_save_df(df, args.hdf5_variable_path, args.stock_name, args.s, args.day, args.month, args.year, "outside_trading", category)

        write_end_time = time.time()

    #Write the time analysis to a text file
    if args.var_analysis_path is not None and args.stock_name == "IBM" and args.day=="03":
        with open(args.var_analysis_path, "a") as f:
            f.write(f"Stock: {args.stock_name}\n")
            f.write(f"\nSuffix: {args.s}\n")
            f.write(f"Day: {args.day}\n")
            f.write(f"Only the calculation runtime: {main_end_time - main_start_time} seconds\n")
            f.write(f"Only the auction processing: {end_auction_time - start_auction_time} seconds\n")
            f.write(f"Only the trade processing: {end_process_trades_time - start_process_trades_time} seconds\n")
            f.write(f"OIB processing: {end_process_OIB_time - start_process_OIB_time} seconds\n")
            f.write(f"Herfindahl Index processing: {end_process_herfindahl_time- start_process_herfindahl_time} seconds\n")
            f.write(f"Only the quote processing: {end_process_quotes_time - start_process_quotes_time} seconds\n")
            f.write(f"Only the midpoint processing: {end_process_midpoint_time - start_process_midpoint_time} seconds\n")
            f.write(f"Only the return processing: {end_process_returns_time - start_process_returns_time} seconds\n")
            f.write(f"Only the variance ratios processing: {end_process_vr_time - start_process_vr_time} seconds\n")
            f.write(f"Write runtime: {write_end_time - write_start_time} seconds\n")

if __name__ == "__main__":
    if args.prof_analysis_path is not None and args.stock_name == "IBM":
        # Profile the main function
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()

        # Save profiling results
        try:
            with open(args.prof_analysis_path, "a") as f:
                f.write(f"\nStock: {args.stock_name}\n")
                f.write(f"\nSuffix: {args.s}\n")
                ps = pstats.Stats(pr, stream=f)
                ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
        except IOError as e:
            print(f"An error occurred while writing the profiling data: {e}")
            raise
    else:
        main()
