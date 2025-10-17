# Required libraries
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import ema_indicator, CCIIndicator, PSARIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
from vnstock import Vnstock, Listing
import os
import time
from datetime import datetime, timedelta
import icb_stock_lists
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image
import pandas_ta as ta

def generate_dsl_buy_signals(df, rsi_length=10, dsl_length=10, zlema_length=10):
    rsi = RSIIndicator(df['close'], window=rsi_length).rsi()

    def zlema(series, length):
        lag = (length - 1) // 2
        ema_data = 2 * series - series.shift(lag)
        return ema_data.ewm(span=length, adjust=False).mean()

    def dsl_lines(series, length, mode):
        up = pd.Series(index=series.index, dtype='float64')
        dn = pd.Series(index=series.index, dtype='float64')
        sma = series.rolling(window=length).mean()
        for i in range(len(series)):
            if i == 0:
                up.iloc[i] = 0
                dn.iloc[i] = 0
            else:
                up.iloc[i] = (
                    up.iloc[i - 1] + mode / length * (series.iloc[i] - up.iloc[i - 1])
                    if series.iloc[i] > sma.iloc[i] else up.iloc[i - 1]
                )
                dn.iloc[i] = (
                    dn.iloc[i - 1] + mode / length * (series.iloc[i] - dn.iloc[i - 1])
                    if series.iloc[i] < sma.iloc[i] else dn.iloc[i - 1]
                )
        return up, dn

    lvlu, lvld = dsl_lines(rsi, dsl_length, 2) # 1 is slow mode, 2 is fast mode
    dsl_osc = zlema((lvlu + lvld) / 2, zlema_length)
    _, level_dn = dsl_lines(dsl_osc, 10, 2) # 1 is slow mode, 2 is fast mode

    buy_signal = (dsl_osc > level_dn) & (dsl_osc.shift(1) < level_dn.shift(1)) & (dsl_osc < 55)
    buy_signal = buy_signal.fillna(False)

    return df.loc[buy_signal].copy()


def compute_trend_trader_remastered(df):
    df = df.copy()
    df = df.reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Compute PSAR and lag
    psar = PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step=0.02, max_step=0.025)
    df['psar'] = psar.psar()
    df['psar_lag'] = df['psar'].shift(1)

    # Detect PSAR lag crossover above high from below
    df['psar_cross_high'] = (df['high'] > df['psar_lag']) & (df['high'].shift(1) <= df['psar_lag'].shift(1))
    df['psar_lag_cross'] = np.where(
        (df['psar_lag'].shift(1) > df['high'].shift(1)) & df['psar_cross_high'], 1, 0
    )

    # Generate BUY signals
    df['buy_signal'] = df['psar_lag_cross'] == 1

    # Extract buy signal rows
    buy_signals = df[df['buy_signal']].copy()

    # Reset index to expose datetime column
    buy_signals.reset_index(inplace=True)

    # Select and reorder columns to match target format
    return buy_signals[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

# Combined strategy logic
def dsl_trendTrader_strategy(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Step 1: Get signal datetimes from both functions
    dsl_signal_df = generate_dsl_buy_signals(df)
    ttr_signal_df = compute_trend_trader_remastered(df)

    dsl_times = pd.to_datetime(dsl_signal_df['datetime']).tolist()
    ttr_times = pd.to_datetime(ttr_signal_df['datetime']).tolist()

    # Step 2: Map datetime to index
    datetime_to_index = {dt: idx for idx, dt in enumerate(df['datetime'])}
    dsl_indices = [datetime_to_index[dt] for dt in dsl_times if dt in datetime_to_index]
    ttr_indices = [datetime_to_index[dt] for dt in ttr_times if dt in datetime_to_index]

    # Step 3: Find all valid pairs within 10 candles
    candidate_indices = []
    for dsl_idx in dsl_indices:
        for ttr_idx in ttr_indices:
            if abs(dsl_idx - ttr_idx) <= 10:
                later_idx = max(dsl_idx, ttr_idx)
                candidate_indices.append(later_idx)

    # Step 4: Remove duplicates and sort
    candidate_indices = sorted(set(candidate_indices))

    # Step 5: Enforce 8-candle spacing between final valid signals
    valid_indices = []
    last_valid_idx = -9
    for idx in candidate_indices:
        if idx - last_valid_idx >= 8:
            valid_indices.append(idx)
            last_valid_idx = idx

    # Step 6: Return valid signals
    df['valid_signal'] = False
    df.loc[valid_indices, 'valid_signal'] = True

    return df[df['valid_signal']]

# === Big Money Detection (from MCDX logic) ===
def detect_big_money(df):
    """
    Adds MCDX-style smart money metrics to the dataframe.
    No plotting is done here — only calculations.

    Returns
    -------
    df : pd.DataFrame with added columns:
         ['RSI_Banker', 'RSI_HotMoney', 'Banker_MA', 'RSI_Retailer']
    """
    RSIBaseBanker = 50
    RSIPeriodBanker = 50
    RSIBaseHotMoney = 30
    RSIPeriodHotMoney = 40
    SensitivityBanker = 1.5
    SensitivityHotMoney = 0.7
    bankerPeriodMA = 10

    df["_RSI_Banker_raw"] = SensitivityBanker * (ta.rsi(df["close"], RSIPeriodBanker) - RSIBaseBanker)
    df["_RSI_HotMoney_raw"] = SensitivityHotMoney * (ta.rsi(df["close"], RSIPeriodHotMoney) - RSIBaseHotMoney)

    df["RSI_Banker"] = df["_RSI_Banker_raw"].clip(0, 20)
    df["RSI_HotMoney"] = df["_RSI_HotMoney_raw"].clip(0, 20)
    df["Banker_MA"] = df["RSI_Banker"].rolling(bankerPeriodMA).mean()
    df["RSI_Retailer"] = 20 - df["RSI_HotMoney"]

    return df

