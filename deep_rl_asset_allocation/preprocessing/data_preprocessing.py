"""Set of functions to pre-process raw csv data for Dow Jones.

Data is pulled from Compustat database via Wharton Research Data Services.
"""

import datetime

import numpy as np
import pandas as pd
from deep_rl_asset_allocation.configs import data_config, paths_config
from stockstats import StockDataFrame


def preprocess_djia_data(filename: str = paths_config.TRAINING_DATA_FILE) -> pd.DataFrame:
    """data preprocessing pipeline for training data"""
    df = _load_dataset(filename)
    # Format date column
    df["Date"] = df.apply(convert_datadate_to_datetime, axis=1)
    # Filter for data at and after training start date
    df = df[df.Date >= data_config.TRAINING_START]
    # add technical indicators using stockstats
    print("Adding Technical Indicators ..")
    df_tech_indicators = _add_technical_indicator(df)
    # add turbulence
    print("Adding Turbulence ..")
    df_final = _add_turbulence(df_tech_indicators)
    # fill the missing values at the beginning - backward fill
    df_final.fillna(method='bfill', inplace=True)
    df["Date"] = df.apply(convert_datadate_to_datetime, axis=1)
    return df_final


def _load_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename).reset_index(drop=True)
    return df


def convert_datadate_to_datetime(df):
    Date = str(df['Date'])
    # output is '2009-01-02'
    year, month, day = Date[:4], Date[5:7], Date[8:10]
    return datetime.date(year=int(year), month=int(month), day=int(day))


def _add_technical_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators using stockstats package to add technical inidactors """

    stock = StockDataFrame.retype(df.copy())

    stock['close'] = stock['adj close']
    unique_ticker = stock.ticker.unique()

    # moving average convergence divergence: most commonly used momentum indicator that identifies moving averages
    macd = pd.DataFrame()
    # relative strength index: quantifies the extent of recent price changes
    rsi = pd.DataFrame()
    # commodity channel index: compares current price to average price over a time window to indicate a buying or selling action
    cci = pd.DataFrame()
    # average directional index: identifies trend strength by quantifying the amount of price movement
    dx = pd.DataFrame()

    for i in range(len(unique_ticker)):
        # macd
        temp_macd = stock[stock.ticker == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        # rsi
        temp_rsi = stock[stock.ticker == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        # cci
        temp_cci = stock[stock.ticker == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        # adx
        temp_dx = stock[stock.ticker == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)

    df['MACD'] = macd
    df['RSI'] = rsi
    df['CCI'] = cci
    df['ADX'] = dx

    return df


def _add_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """add turbulence index from a precalculated dataframe"""
    turbulence_index = _calculate_turbulence(df)
    df = df.merge(turbulence_index, on='Date')
    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    return df


def _calculate_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    df_price_pivot = df.pivot(index='Date', columns='Ticker', values='Adj Close')
    unique_date = df.Date.unique()
    # start after a year
    start = 1 * data_config.NUMBER_OF_TRADING_DAYS_PER_YEAR
    turbulence_index = [0] * start
    count = 0
    for idx in range(start, len(unique_date)):
        # stock returns for current period t
        current_price = df_price_pivot[df_price_pivot.index == unique_date[idx]]
        # average of historical returns
        hist_price = df_price_pivot[[n in unique_date[0:idx] for n in df_price_pivot.index]]
        # covariance of historical returns
        cov_temp = hist_price.cov()
        # calculate turbulence
        current_temp = (current_price - np.mean(hist_price, axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)
    turbulence_index = pd.DataFrame({'Date': df_price_pivot.index, 'Turbulence': turbulence_index})
    return turbulence_index
