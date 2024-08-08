# aggregation_functions.py
import pandas as pd
import numpy as np
import polars as pl
from numba import njit

#Separate opening and closing prices from the variable calculation, will be saved in daily auction group
def auction_conditions(pl_df):
    special_conditions_df = pl_df.filter(pl.col('cond').str.contains('M|O|P|Q|5|6|9'))     #these codes correspond to opening and closing prices 
    cleaned_df = pl_df.filter(~pl.col('cond').str.contains('M|O|P|Q|5|6|9'))               #now remove opening and closing prices from trades dataframe
    special_conditions_pd = special_conditions_df.select(['time', 'price', 'vol', 'EX', 'cond']).to_pandas()
    cleaned_df_pd = cleaned_df.to_pandas()
    return special_conditions_pd, cleaned_df_pd

#Fill the minutes in the time column after the time bars are calculated for different dataframes and variables, so that entire time bar data can be merged in one dataframe
def reindex_to_full_time(df, base_date, outside_trading=False):
    if df is None or df.empty or df.isna().all().all():
        return None
    
    if outside_trading:
        morning_index = pd.date_range(start=f"{base_date} 03:30", end=f"{base_date} 09:29", freq="30min")
        evening_index = pd.date_range(start=f"{base_date} 16:00", end=f"{base_date} 22:00", freq="30min")
        full_time_index = morning_index.union(evening_index)
    else:
        full_time_index = pd.date_range(start=f"{base_date} 09:30", end=f"{base_date} 15:59", freq="1min")
    df_reindexed = df.reindex(full_time_index)
    return df_reindexed

def reindex_to_seconds(df, base_date):
    full_time_index = pd.date_range(start=f"{base_date} 09:30", end=f"{base_date} 15:59", freq="1s")
    df_reindexed = df.reindex(full_time_index).fillna(0)
    return df_reindexed

#Calculate the variance

def calculate_minute_variance(returns):
    x = returns.to_numpy()
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= 1:
        return np.nan
    return x.var()

#Calculate the volatility

def calculate_minute_volatility(series):
    x = series.to_numpy()
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= 1:
        return np.nan
    return x.std()

#Calculate the partial autocorrelation
def calculate_autocorrelation_v1(series, lag=1):
    x = series.to_numpy()
    x = x[~np.isnan(x)]
    if len(x) <= lag:
        return np.nan
    if np.var(x) == 0 or np.var(x[:-lag]) == 0 or np.var(x[lag:]) == 0:
        return np.nan
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


#Calculate the non-partial autocorrelation

def calculate_autocorrelation_v2(series, lag=1):
    x = series.to_numpy()
    x = x[~np.isnan(x)]
    if len(x) < lag:
        return np.nan
    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    if var == 0:
        return np.nan
    corr=np.sum(xp[lag:]*xp[:-lag])/len(x)/var
    return np.array(corr)


#Calculate the orderflow by Chordia, Hu, Subrahmanyam and Tong, MS 2019

def calculate_oib_metrics(df1_filtered, df2_filtered, base_date):
    if df1_filtered is None or df1_filtered.empty or df1_filtered.isna().all().all() or df2_filtered is None or df2_filtered.empty or df2_filtered.isna().all().all():
        return None
    
    df1_filtered.reset_index(inplace = True)
    df2_filtered.reset_index(inplace = True)

    df1_pl = pl.from_pandas(df1_filtered)
    df2_pl = pl.from_pandas(df2_filtered)

    aggregations = [
        pl.col('vol').sum().alias('shr'),
        pl.col('vol').count().alias('num'),
        pl.col('value').sum().alias('doll'),
    ]
    
    buys_per_s = df1_pl.group_by_dynamic('time', every='1s', closed='left', label='left').agg(aggregations)
    sells_per_s = df2_pl.group_by_dynamic('time', every='1s', closed='left', label='left').agg(aggregations)

    buys_per_s = buys_per_s.to_pandas().set_index("time")
    sells_per_s = sells_per_s.to_pandas().set_index("time")
    buys_per_s = reindex_to_seconds(buys_per_s, base_date)
    sells_per_s = reindex_to_seconds(sells_per_s, base_date)

    oib_shr_s = (buys_per_s['shr'] - sells_per_s['shr']) / (buys_per_s['shr'] + sells_per_s['shr'])
    oib_num_s = (buys_per_s['num'] - sells_per_s['num']) / (buys_per_s['num'] + sells_per_s['num'])
    oib_doll_s = (buys_per_s['doll'] - sells_per_s['doll']) / (buys_per_s['doll'] + sells_per_s['doll'])
    
    oib_metrics = pd.DataFrame({
        'OIB_SHR': oib_shr_s,
        'OIB_NUM': oib_num_s,
        'OIB_DOLL': oib_doll_s
    })
    oib_metrics.dropna(inplace=True)
    oib_metrics.reset_index(inplace = True)
    oib_metrics.rename(columns={'index': 'time'}, inplace=True)
    return oib_metrics

#Calculate the variance and autocorrelation of orderflow by Chordia, Hu, Subrahmanyam and Tong, MS 2019
def apply_oib_aggregations(df):
    if df is None or df.empty or df.isna().all().all():
        return None
    
    pl_df = pl.from_pandas(df)
    if pl_df.height == 0 or pl_df.height == 1:
        return None

    resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg([
        pl.map_groups(
            exprs=["OIB_SHR"],
            function=lambda groups: calculate_minute_volatility(groups[0]),
            return_dtype=pl.Float64
        ).alias('OIB_SHR_volatility_s'),
        pl.map_groups(
            exprs=["OIB_SHR"],
            function=lambda groups: calculate_autocorrelation_v1(groups[0]),
            return_dtype=pl.Float64
        ).alias('OIB_SHR_autocorr_s'),
        pl.map_groups(
            exprs=["OIB_NUM"],
            function=lambda groups: calculate_minute_volatility(groups[0]),
            return_dtype=pl.Float64
        ).alias('OIB_NUM_volatility_s'),
        pl.map_groups(
            exprs=["OIB_NUM"],
            function=lambda groups: calculate_autocorrelation_v1(groups[0]),
            return_dtype=pl.Float64
        ).alias('OIB_NUM_autocorr_s'),
        pl.map_groups(
            exprs=["OIB_DOLL"],
            function=lambda groups: calculate_minute_volatility(groups[0]),
            return_dtype=pl.Float64
        ).alias('OIB_DOLL_volatility_s'),
        pl.map_groups(
            exprs=["OIB_DOLL"],
            function=lambda groups: calculate_autocorrelation_v1(groups[0]),
            return_dtype=pl.Float64
        ).alias('OIB_DOLL_autocorr_s')
    ])
    return resampled_df.to_pandas().set_index('time')
        
#Calculate the variance and autocorrelation of returns, receives a polars dataframe
def apply_return_aggregations(pl_df, column='returns', sign=False, outside_trading=False):
    if pl_df is None or pl_df.shape[0] == 0:
        return None
    if pl_df.shape[0] == 1:
        return None

    if sign is False:
        volatility_col_name = 'ret_volatility_s' if not outside_trading else 'ret_volatility_m'
        autocorr_col_name = 'ret_autocorr_s' if not outside_trading else 'ret_autocorr_m'
    else:
        volatility_col_name = f'sign_volatility_s' if not outside_trading else f'sign_volatility_m'
        autocorr_col_name = f'sign_autocorr_s' if not outside_trading else f'sign_autocorr_m'

    interval = '30m' if outside_trading else '1m'

    resampled_df = pl_df.group_by_dynamic('time', every=interval, closed='left', label='left').agg([
        pl.map_groups(
            exprs=[column],
            function=lambda groups: calculate_minute_volatility(groups[0]),
            return_dtype=pl.Float64
        ).alias(volatility_col_name),
        pl.map_groups(
            exprs=[column],
            function=lambda groups: calculate_autocorrelation_v1(groups[0]),
            return_dtype=pl.Float64
        ).alias(autocorr_col_name)
    ])
    return resampled_df.to_pandas().set_index('time')

    #Calculate only the variance of returns
def apply_ret_variances_aggregations(pl_df, name, column='returns'):
    if pl_df is None or pl_df.shape[0] == 0:
        return None
    if pl_df.shape[0] == 1:
        return None
    resampled_df = pl_df.group_by_dynamic('time', every='1m', closed='left', label='left').agg([
        pl.map_groups(
            exprs=[column],
            function=lambda groups: calculate_minute_variance(groups[0]),
            return_dtype=pl.Float64
        ).alias(name)
    ])
    return resampled_df.to_pandas().set_index('time')


# Aggregated Functions for Trades
#Variables for the trades dataframes, most of them are formed at the same time 
def apply_aggregations(df_filtered, df_name, outside_trading=False):
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
    interval_seconds = '1s' if not outside_trading else '1m'
    interval_minutes = '1m' if not outside_trading else '30m'
    max_events_label = 'max_events_s' if not outside_trading else 'max_events_m'

    try:
        #Group the dataframe to 1-second intervals and count the events inside
        seconds_df = pl_df.group_by_dynamic('time', every=interval_seconds, closed='left', label='left').agg([
            pl.count('price').alias('count')
        ])

        #Compute the variable: maximum events in the seconds per minute
        max_trades = seconds_df.group_by_dynamic('time', every=interval_minutes, closed='left', label='left').agg([
            pl.col('count').max().alias(max_events_label)
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
        if df_name in ['Buys_trades', 'Buys_Oddlot_trades', 'Buys_Retail_trades']:
            aggregations.append(pl.col('pNextSell_tob').mean().alias('pNextSell_avg'))
            aggregations.append(pl.col('dtNextSell_tob').mean().alias('dtNextSell_avg'))

        if df_name in ['Sells_trades', 'Sells_Oddlot_trades', 'Sells_Retail_trades']:
            aggregations.append(pl.col('pNextBuy_tos').mean().alias('pNextBuy_avg'))
            aggregations.append(pl.col('dtNextBuy_tos').mean().alias('dtNextBuy_avg'))

        #Resample to one minute bars, using the aggregations above
        resampled_df = pl_df.group_by_dynamic('time', every=interval_minutes, closed='left', label='left').agg(aggregations)
        resampled_df = resampled_df.to_pandas()
        max_trades = max_trades.to_pandas()
        #Merge with the aggregation of maximum number of trades which was performed prior to the others
        resampled_df = resampled_df.merge(max_trades, on='time', how='left')
        #Return the time bar variables, set the time as an index
        return resampled_df.set_index('time')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#Aggregated Functions for Quotes
def apply_quote_aggregations(df_filtered, df_name, outside_trading=False):
    if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
        return None
    if len(df_filtered) == 1:
        return df_filtered

    df_filtered.reset_index(inplace = True)

    #Calculate durations of prices and weighted_price by these durations
    df_filtered['durations'] = df_filtered['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
    df_filtered['weighted_price'] = df_filtered['price'] * df_filtered['durations']

    pl_df = pl.from_pandas(df_filtered)
    interval_seconds = '1s' if not outside_trading else '1m'
    interval_minutes = '1m' if not outside_trading else '30m'
    max_events_label = 'max_events_s' if not outside_trading else 'max_events_m'

    try:
        #Group the dataframe to 1-second intervals and count the events inside
        seconds_df = pl_df.group_by_dynamic('time', every=interval_seconds, closed='left', label='left').agg([
            pl.count('price').alias('count')
        ])

        #Compute the variable: maximum events in the seconds per minute
        max_trades = seconds_df.group_by_dynamic('time', every=interval_minutes, closed='left', label='left').agg([
            pl.col('count').max().alias(max_events_label)
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

        resampled_df = pl_df.group_by_dynamic('time', every=interval_minutes, closed='left', label='left').agg(aggregations)

        resampled_df = resampled_df.to_pandas()
        max_trades = max_trades.to_pandas()
        #Merge with the aggregation of maximum number of trades which was performed prior to the others
        resampled_df = resampled_df.merge(max_trades, on='time', how='left')
        #Return the time bar variables, set the time as an index
        return resampled_df.set_index('time')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#Functions for Midprice

def apply_midpoint_aggregations(df_filtered, outside_trading=False):
    if df_filtered is None or df_filtered.empty or df_filtered.isna().all().all():
        return None
    if len(df_filtered) == 1:
        return df_filtered
    
    pl_df = pl.from_pandas(df_filtered.reset_index())
    interval_seconds = '1s' if not outside_trading else '1m'
    interval_minutes = '1m' if not outside_trading else '30m'
    max_events_label = 'max_events_s' if not outside_trading else 'max_events_m'

    try:
        #Group the dataframe to 1-second intervals and count the events inside
        seconds_df = pl_df.group_by_dynamic('time', every=interval_seconds, closed='left', label='left').agg([
            pl.count('price').alias('count')
        ])

        #Compute the variable: maximum events in the seconds per minute
        max_trades = seconds_df.group_by_dynamic('time', every=interval_minutes, closed='left', label='left').agg([
            pl.col('count').max().alias(max_events_label)
        ])

        #Count the number of events
        aggregations = [
            pl.count('price').alias('num_events'),
        ]

        resampled_df = pl_df.group_by_dynamic('time', every=interval_minutes, closed='left', label='left').agg(aggregations)
        resampled_df = resampled_df.to_pandas()
        max_trades = max_trades.to_pandas()
        #Merge with the aggregation of maximum number of trades which was performed prior to the others
        resampled_df = resampled_df.merge(max_trades, on='time', how='left')
        #Return the time bar variables, set the time as an index
        return resampled_df.set_index('time')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
#For resampling returns, outputs a polars dataframe
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
    resampled_df = pl_df.group_by_dynamic('time', every=interval, closed='left', label='left').agg(aggregations)

    if resampled_df.is_empty():
        return None
    #since the volume has been used for the weighting, it is now dropped from the dataframe
    if 'vol' in resampled_df.columns:
        resampled_df = resampled_df.drop('vol')
    return resampled_df

def process_daily(df_filtered_inside, df_filtered_outside, set, column, is_cond=True):
    def calculate_vwap(df):
        total_volume = df['vol'].sum()
        if total_volume == 0:
            return 0.0
        return (df['price'] * df['vol']).sum() / total_volume

    # Dictionaries to store the results
    daily_inside = {}
    daily_outside = {}

    # Loop through both intervals: inside and outside
    for interval_name, df_interval in zip(['inside', 'outside'], [df_filtered_inside, df_filtered_outside]):
        # Loop through characters in the set (condition or exchange)
        for char in set:
            if char == '@':
                key = 'cond_at' if is_cond else 'ex_at'
            elif char == '':
                key = 'cond_empty' if is_cond else 'ex_empty'
            else:
                key = f'cond_{char}' if is_cond else f'ex_{char}'

            if char == '' and is_cond:
                df_filtered = df_interval.filter(pl.col(column) == '')
            else:
                df_filtered = df_interval.filter(pl.col(column).str.contains(char))

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
        #compute one time, e.g. when the function is called for conditions, the metrics for all trades inside and seperately outside trading hours
        if is_cond == True:
            if interval_name == 'inside':
                daily_inside['vwap'] = calculate_vwap(df_interval)
                daily_inside['tot_vol'] = df_interval['vol'].sum()
                daily_inside['no_buys'] = df_interval.filter(pl.col('Initiator') == 1).height
                daily_inside['no_sells'] = df_interval.filter(pl.col('Initiator') == -1).height
            else:
                daily_outside['vwap'] = calculate_vwap(df_interval)
                daily_outside['total_vol'] = df_interval['vol'].sum()
                daily_outside['no_buys'] = df_interval.filter(pl.col('Initiator') == 1).height
                daily_outside['no_sells'] = df_interval.filter(pl.col('Initiator') == -1).height

    return daily_inside, daily_outside

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_Herfindahl(df):
    if df is None or df.empty or df.isna().all().all():
        return None
    if 'time' in df.columns:
        df.set_index('time', inplace=True)

    df_filtered = df.between_time("09:30", "15:59:59").copy()
    if df_filtered.empty or df_filtered.isna().all().all():
        return None
    
    pl_df = pl.from_pandas(df_filtered.reset_index())
    
    resampled = pl_df.group_by_dynamic('time', every='1s', closed='left', label='left').agg([
        pl.col('value').sum().alias('sum'),
        (pl.col('value')**2).sum().alias('sum_of_squared')
    ])
    
    minutely_data = resampled.group_by_dynamic('time', every='1m', closed='left', label='left').agg([
        pl.col('sum').sum().alias('double_sum'),
        pl.col('sum_of_squared').sum().alias('double_sum_of_squared')
    ])
    minutely_data = minutely_data.with_columns([
        (minutely_data['double_sum']**2).alias('sq_double_sum')
    ])
    
    minutely_data = minutely_data.with_columns([
        (minutely_data['double_sum_of_squared'] / minutely_data['sq_double_sum']).alias('Herfindahl_s')
    ])
    minutely_data = minutely_data.select([
        'time', 'Herfindahl_s'
    ])
    minutely_data = minutely_data.to_pandas().set_index('time')
    return minutely_data