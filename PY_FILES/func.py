
import re
import ta
import os
import atexit
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import entropy
from datetime import datetime, timedelta, timezone
import warnings
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", InconsistentVersionWarning)
warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)

# def info_init():
#     url = "https://trying-20541-default-rtdb.firebaseio.com/Main_info.json"
#     response = requests.get(url)
#     data = response.json()['main_init']
#     LOGGER.info(data)
# info_init()

LOGGER = None

def set_logger(logger):
    global LOGGER
    LOGGER = logger

def apply_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"])
    df["High"] = pd.to_numeric(df["High"])
    df["Low"] = pd.to_numeric(df["Low"])
    df["Open"] = pd.to_numeric(df["Open"])
    df["Volume"] = pd.to_numeric(df["Volume"])

    df['Hour'] = df['Date'].dt.hour
    df['london_session'] = ((df['Hour'] >= 7) & (df['Hour'] <= 16)).astype(int)
    df['ny_session'] = ((df['Hour'] >= 13) & (df['Hour'] <= 22)).astype(int)
    df['asia_session'] = ((df['Hour'] >= 0) & (df['Hour'] <= 8)).astype(int)
    df['Weekday'] = df['Date'].dt.weekday

    vol_change = df['Volume'].pct_change()
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=15)
    df['candle_dir'] = (df['Close'] > df['Open']).astype(int)

    for WINDOW in [10, 14, 20, 50, 100, 200]:
        df[f'EMA_{WINDOW}'] = ta.trend.ema_indicator(df['Close'], window=WINDOW)
        df[f'SMA_{WINDOW}'] = ta.trend.sma_indicator(df['Close'], window=WINDOW)
        df[f'WMA_{WINDOW}'] = ta.trend.wma_indicator(df['Close'], window=WINDOW)
        df[f'RSI_{WINDOW}'] = ta.momentum.rsi(df['Close'], window=WINDOW)
        df[f'ATR_{WINDOW}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=WINDOW)
        df[f'ADX_{WINDOW}'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=WINDOW)
        df[f'KAMA_{WINDOW}'] = ta.momentum.KAMAIndicator(close=df['Close'], window=WINDOW).kama()
        df[f'Volume_change_{WINDOW}'] = vol_change.rolling(WINDOW).mean()
        df[f"ATR_{WINDOW}"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=WINDOW)

        bb = ta.volatility.BollingerBands(df["Close"], window=WINDOW)
        df[f"BB_H_{WINDOW}"] = bb.bollinger_hband()
        df[f"BB_L_{WINDOW}"] = bb.bollinger_lband() 
        df[f"bollinger_width_{WINDOW}"] = df[f"BB_H_{WINDOW}"] - df[f"BB_L_{WINDOW}"]

        ema = ta.trend.EMAIndicator(close=df['Close'], window=WINDOW).ema_indicator()
        ema_of_ema = ta.trend.EMAIndicator(close=ema, window=WINDOW).ema_indicator()
        df[f'DEMA_{WINDOW}'] = 2 * ema - ema_of_ema
        ema2 = ema_of_ema
        ema3 = ta.trend.EMAIndicator(close=ema2, window=WINDOW).ema_indicator()
        df[f'TEMA_{WINDOW}'] = 3 * ema - 3 * ema2 + ema3

        df[f'ATR_MEAN_{WINDOW}'] = df['ATR'].rolling(WINDOW).mean()
        df[f'dist_to_prev_high_{WINDOW}'] = (df['Close'] - df['High'].rolling(WINDOW).max()) / df['ATR']
        df[f'dist_to_prev_low_{WINDOW}'] = (df['Close'] - df['Low'].rolling(WINDOW).min()) / df['ATR']
    
        vol_mean = df['Volume'].rolling(WINDOW).mean()
        vol_std  = df['Volume'].rolling(WINDOW).std()
        df[f'volume_spike_{WINDOW}'] = (df['Volume'] > (vol_mean + vol_std)).astype(int)
        df[f'vol_candle_signal_{WINDOW}'] = ((df['candle_dir'] == 1) &(df[f'volume_spike_{WINDOW}'] == 1)).astype(int)

        df[f'Dist_to_Recent_High_{WINDOW}'] = (df['High'].rolling(WINDOW).max() - df['Close']) / df['ATR']
        df[f'Dist_to_Recent_Low_{WINDOW}'] = (df['Close'] - df['Low'].rolling(WINDOW).min()) / df['ATR']
        df[f'Dist_to_Rolling_Max_{WINDOW}'] = (df['Close'].rolling(WINDOW).max() - df['Close']) / df['ATR']
        df[f'Dist_to_Rolling_Min_{WINDOW}'] = (df['Close'] - df['Close'].rolling(WINDOW).min()) / df['ATR']
        df[f'Rolling_Mean_Volume_{WINDOW}'] = df["Volume"].rolling(window=WINDOW).mean()
        df[f'Volume_Spike_{WINDOW}'] = df["Volume"] / df[f'Rolling_Mean_Volume_{WINDOW}']
        df[f'Volume_Z_{WINDOW}'] = ((df['Volume'] - df[f'Rolling_Mean_Volume_{WINDOW}']) /df['Volume'].rolling(WINDOW).std()).clip(-5, 5)
        
        df[f'Bull_Pressure_{WINDOW}'] = ((df['Close'] > df['Open']).rolling(WINDOW).mean())
        df[f'Bear_Pressure_{WINDOW}'] = ((df['Close'] < df['Open']).rolling(WINDOW).mean())
  
        df[f'Price_Change_{WINDOW}'] = df['Close'].diff()
        df[f'Divergence_{WINDOW}'] = np.where((df[f'RSI_{WINDOW}'].diff() > 0) & (df[f'Price_Change_{WINDOW}'] < 0), 1, 0)
        df[f'Overbought_{WINDOW}'] = np.where(df[f'RSI_{WINDOW}'] > 70, 1, 0)
        df[f'Oversold_{WINDOW}'] = np.where(df[f'RSI_{WINDOW}'] < 30, 1, 0)

        df[f'ATR_percentile_{WINDOW}'] = (df['ATR'].rolling(WINDOW).rank(pct=True))
        df[f'High_Volatility_{WINDOW}'] = (df[f'ATR_percentile_{WINDOW}'] > 0.7).astype(int)

        bb_mean = df[f'bollinger_width_{WINDOW}'].rolling(WINDOW).mean()
        bb_std = df[f'bollinger_width_{WINDOW}'].rolling(WINDOW).std()
        df[f'Compression_State_{WINDOW}'] = (df[f'bollinger_width_{WINDOW}'] < (bb_mean - 0.5 * bb_std)).astype(int)

        df[f'Expansion_State_{WINDOW}'] = (
            (df[f'bollinger_width_{WINDOW}'] > bb_mean + 0.5 * bb_std) &(df[f'ATR_{WINDOW}'] > df[f'ATR_{WINDOW}'].rolling(WINDOW).mean())).astype(int)


    df["ATR_ratio_8_14"] = df["ATR"] / df["ATR_14"]
    df["ATR_ratio_14_49"] = df["ATR_14"] / df["ATR_50"]
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-6)
    df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])
    df['TREND'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)
    df['trend_strength'] = (df['EMA_50'] - df['EMA_200']) / df['Close']
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_signal"] = ta.trend.macd_signal(df["Close"])
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])

    df["+DI"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"])
    df["-DI"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"])
    df["Ichimoku_A"] = ta.trend.ichimoku_a(df["High"], df["Low"])
    df["Ichimoku_B"] = ta.trend.ichimoku_b(df["High"], df["Low"])
    df["TRIX"] = ta.trend.trix(df["Close"])
    df["KST"] = ta.trend.kst(df["Close"])
    df["Stoch_RSI"] = ta.momentum.stochrsi(df["Close"])
    df["ROC"] = ta.momentum.roc(df["Close"])
    df["AO"] = ta.momentum.awesome_oscillator(df["High"], df["Low"])
    df["Donchian_high"] = ta.volatility.donchian_channel_hband(df["High"], df["Low"], df["Close"])
    df["Donchian_low"] = ta.volatility.donchian_channel_lband(df["High"], df["Low"], df["Close"])
    df["Keltner_high"] = ta.volatility.keltner_channel_hband(df["High"], df["Low"], df["Close"])
    df["Keltner_low"] = ta.volatility.keltner_channel_lband(df["High"], df["Low"], df["Close"])
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["MFI"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"])
    df["AD"] = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])

    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['pivot_high'] = ((df['High'] > df['High'].shift(1)) &(df['High'] > df['High'].shift(2))).shift(2).fillna(0).astype(int)
    df['pivot_low'] = (((df['Low'] < df['Low'].shift(1)) &df['Low'] < df['Low'].shift(2))).shift(2).fillna(0).astype(int)
    df['slope_ema10'] = (df['EMA_10'] - df['EMA_10'].shift(1)) / df['Close']
    df['ema10_20_state'] = (df['EMA_10'] > df['EMA_20']).astype(int)
    df['ma_alignment'] = ((df['EMA_10'] > df['EMA_20']) &(df['EMA_20'] > df['EMA_50'])).astype(int)
    df['price_diff'] = (df['Close'] - df['Close'].shift(1)) / df['Close']
    df['rsi_diff'] = df['RSI_20'] - df['RSI_50'].shift(1)
    df['Dist_from_EMA200'] = (df['Close'] - df['EMA_200']) / df['ATR']
    df['divergence_flag'] = (
        (df['price_diff'].rolling(3).mean() > 0) &
        (df['rsi_diff'].rolling(3).mean() < 0)) | ((df['price_diff'].rolling(3).mean() < 0) &(df['rsi_diff'].rolling(3).mean() > 0))
    df['divergence_flag'] = df['divergence_flag'].astype(int)
    df['candle_body'] = abs(df['Close'] - df['Open']) / df['Close']

    df['Body_to_Range'] = (abs(df['Close'] - df['Open']) /((df['High'] - df['Low']) + 1e-6))
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Mean_Return'] = df['Log_Return'].rolling(5).mean()
    df['Rolling_Std_Return'] = df['Log_Return'].rolling(5).std()
    df['EMA_20_50_dist'] = (df['EMA_20'] - df['EMA_50']) / df['ATR']
    df['Dist_from_EMA5'] = (df['Close'] - df['EMA_10']) / df['ATR']
    df['Trend_Strength'] = abs(df['Close'] - df['EMA_10']) / df['ATR']
    df["Vol_Range"] = df["Volume"] * (df["High"] - df["Low"])

    # ==========================================================
    # SUPPORT & RESISTANCE LEVELS
    # ==========================================================

    for period in [20, 50, 100, 200]:

        # Support & Resistance
        df[f"support_{period}"] = df["Low"].rolling(period).min()
        df[f"resistance_{period}"] = df["High"].rolling(period).max()

        # ======================================================
        # ATR-NORMALIZED DISTANCES
        # ======================================================

        df[f"dist_support_{period}"] = ((df["Close"] - df[f"support_{period}"])/ df["ATR"])
        df[f"dist_resistance_{period}"] = ((df[f"resistance_{period}"] - df["Close"])/ df["ATR"])

        # ======================================================
        # BREAKOUT FLAGS
        # ======================================================

        df[f"breakout_up_{period}"] = (df["Close"] > df[f"resistance_{period}"]).astype(int)

        df[f"breakout_down_{period}"] = (df["Close"] < df[f"support_{period}"]).astype(int)

        # ======================================================
        # ATR-NORMALIZED BREAKOUT STRENGTH
        # ======================================================

        df[f"breakout_strength_up_{period}"] = ((df["Close"] - df[f"resistance_{period}"])/ df["ATR"])
        df[f"breakout_strength_down_{period}"] = ((df[f"support_{period}"] - df["Close"])/ df["ATR"])


    # ==========================================================
    # MARKET STRUCTURE WINDOWS
    # ==========================================================

    for WINDOW in [20, 50, 100]:

        # ======================================================
        # SUPPORT / RESISTANCE
        # ======================================================

        df[f'rolling_high_{WINDOW}'] = (df['High'].rolling(WINDOW).max().shift(1))

        df[f'rolling_low_{WINDOW}'] = (df['Low'].rolling(WINDOW).min().shift(1))
        df[f'dist_to_resistance_{WINDOW}'] = ((df[f'rolling_high_{WINDOW}'] - df['Close'])/ df['ATR'])

        df[f'dist_to_support_{WINDOW}'] = ((df['Close'] - df[f'rolling_low_{WINDOW}'])/ df['ATR'])
        df[f'near_resistance_{WINDOW}'] = (df[f'dist_to_resistance_{WINDOW}'] < 0.5).astype(int)

        df[f'near_support_{WINDOW}'] = (df[f'dist_to_support_{WINDOW}'] < 0.5).astype(int)

        # ======================================================
        # TURTLE SOUP
        # ======================================================

        df[f'prev_high_{WINDOW}'] = (df['High'].shift(1).rolling(WINDOW).max())

        df[f'prev_low_{WINDOW}'] = (df['Low'].shift(1).rolling(WINDOW).min())

        df[f'broke_prev_high_{WINDOW}'] = (df['High'] > df[f'prev_high_{WINDOW}']).astype(int)

        df[f'broke_prev_low_{WINDOW}'] = (df['Low'] < df[f'prev_low_{WINDOW}']).astype(int)

        df[f'turtle_soup_sell_{WINDOW}'] = ((df['High'] > df[f'prev_high_{WINDOW}']) &(df['Close'] < df[f'prev_high_{WINDOW}'])).astype(int)

        df[f'turtle_soup_buy_{WINDOW}'] = ((df['Low'] < df[f'prev_low_{WINDOW}']) &(df['Close'] > df[f'prev_low_{WINDOW}'])).astype(int)

        # ======================================================
        # BREAK OF STRUCTURE (BOS)
        # ======================================================

        df[f'structure_high_{WINDOW}'] = (df['High'].rolling(WINDOW).max().shift(1))
        df[f'structure_low_{WINDOW}'] = (df['Low'].rolling(WINDOW).min().shift(1))

        df[f'bos_up_{WINDOW}'] = (df['Close'] > df[f'structure_high_{WINDOW}']).astype(int)

        df[f'bos_down_{WINDOW}'] = (df['Close'] < df[f'structure_low_{WINDOW}']).astype(int)

        df[f'structure_direction_{WINDOW}'] = np.where(df[f'bos_up_{WINDOW}'] == 1,1,np.where(df[f'bos_down_{WINDOW}'] == 1,-1,0))

        # ======================================================
        # FIBONACCI
        # ======================================================

        df[f'swing_high_{WINDOW}'] = (df['High'].rolling(WINDOW).max().shift(1))
        df[f'swing_low_{WINDOW}'] = (df['Low'].rolling(WINDOW).min().shift(1))

        fib_range = (df[f'swing_high_{WINDOW}']-df[f'swing_low_{WINDOW}'])
        df[f'fib_38_{WINDOW}'] = (df[f'swing_high_{WINDOW}']-(0.382 * fib_range))

        df[f'fib_50_{WINDOW}'] = (df[f'swing_high_{WINDOW}']-(0.500 * fib_range))
        df[f'fib_618_{WINDOW}'] = (df[f'swing_high_{WINDOW}']-(0.618 * fib_range))

        df[f'dist_fib_618_{WINDOW}'] = (np.abs( df['Close']-df[f'fib_618_{WINDOW}'])/df['ATR'])
        df[f'fib_618_zone_{WINDOW}'] = (df[f'dist_fib_618_{WINDOW}'] < 0.5).astype(int)
        df[f'vol_of_vol_{WINDOW}'] = df['ATR'].rolling(WINDOW).std()

    df['Range'] = df['High'] - df['Low'] + 1e-6
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Candle_Strength'] = (df['Close'] - df['Open']) / df['Range']
    df['Close_Position'] = (df['Close'] - df['Low']) / df['Range']
    df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']
    df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']
    df['wick_ratio'] = df['Range'] / (df['Body'] + 1e-6)
    df['PinBar_Bull'] = ((df['Lower_Wick'] > 0.6) &(df['Upper_Wick'] < 0.2)).astype(int)
    df['PinBar_Bear'] = ((df['Upper_Wick'] > 0.6) &(df['Lower_Wick'] < 0.2)).astype(int)
    df['Impulse_Bull'] = ((df['Body'] / df['Range'] > 0.6) &(df['Close'] > df['Open'])).astype(int)
    df['Impulse_Bear'] = ((df['Body'] / df['Range'] > 0.6) &(df['Close'] < df['Open'])).astype(int)
    df['Inside_Bar'] = ((df['High'] < df['High'].shift(1)) &(df['Low'] > df['Low'].shift(1))).astype(int)
    df['Bull_Engulf'] = ((df['Close'] > df['Open']) &(df['Open'] < df['Close'].shift(1)) &(df['Close'] > df['Open'].shift(1))).astype(int)
    df['Bear_Engulf'] = ((df['Close'] < df['Open']) &(df['Open'] > df['Close'].shift(1)) &(df['Close'] < df['Open'].shift(1))).astype(int)
    df['Doji'] = (df['Body'] / df['Range'] < 0.1).astype(int)

    PRICE_BUCKETS = 50
    range_width = df['ATR_14'] * 3
    price_min = df['Close'] - range_width
    price_max = df['Close'] + range_width
    vol_bucket = ((df['Close'] - price_min) / (price_max - price_min + 1e-6)) * (PRICE_BUCKETS - 1)
    vol_bucket = vol_bucket.replace([np.inf, -np.inf], np.nan)
    vol_bucket = vol_bucket.fillna(0)
    vol_bucket = vol_bucket.clip(0, PRICE_BUCKETS - 1)
    df['vol_bucket'] = vol_bucket.astype(int)

    df['dist_to_high_vol_node'] = df['High'].rolling(40).max() - df['Close']
    df['tick_imbalance'] = (df['Close'] - df['Open']) * df['Volume']
    df['rolling_buy_pressure'] = df['tick_imbalance'].clip(lower=0).rolling(10).sum()
    df['rolling_sell_pressure'] = -df['tick_imbalance'].clip(upper=0).rolling(10).sum()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['rolling_vol'] = df['log_return'].rolling(10).std()
    long_term_vol = df['log_return'].rolling(50).std()
    df['volatility_state'] = (df['rolling_vol'] > long_term_vol).astype(int)
    df['volatility_ratio'] = df['rolling_vol'] / (long_term_vol + 1e-6)
    df['trend'] = (df['EMA_50'] > df['EMA_200']).astype(int)
    df['EMA_200_8_ROLL'] = df['EMA_200'].rolling(8).mean()

    df['candle_range'] = df['High'] - df['Low'] + 1e-6
    df['wick_to_body_ratio'] = ((df['High'] - df[['Open','Close']].max(axis=1)) + 
                                (df[['Open','Close']].min(axis=1) - df['Low'])) / (abs(df['Close'] - df['Open']) + 1e-6)
    df['candle_range_ratio'] = df['candle_range'] / (df['ATR'] + 1e-6)
    df['open_close_ratio'] = abs(df['Open'] - df['Close']) / (df['candle_range'] + 1e-6)
    df['gap_from_prev_close'] = (df['Open'] - df['Close'].shift(1)) / (df['ATR'].shift(1) + 1e-6)

    lookback = 14
    df['fractal_bull'] = ((df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(2))).astype(int)
    df['fractal_bear'] = ((df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(2))).astype(int)
    df['fractal_score'] = df['fractal_bull'] - df['fractal_bear']
    df['fractal_score_smooth'] = df['fractal_score'].rolling(lookback).mean().fillna(0)
    df['trend_slope'] = (df['Close'] - df['Close'].shift(lookback)) / lookback
    df['vol_adj_trend'] = df['trend_slope'] / (df['ATR'] + 1e-6)  # normalized by ATR
    
    window_entropy = 20
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    def rolling_entropy(x):
        hist, _ = np.histogram(x, bins=10, density=True)
        return entropy(hist + 1e-8)  # small value to avoid log(0)
    df['entropy'] = log_ret.rolling(window_entropy).apply(rolling_entropy, raw=True).fillna(0)
    df['vol_regime'] = (df['ATR'] > df['ATR'].rolling(50).mean()).astype(int)
    
    ema_period = 50
    slope_period = 5
    df['EMA_50_SLOPE_ATR'] = (df['Close'].ewm(span=ema_period, adjust=False).mean())
    df['EMA_SLOPE_ATR'] = ((df['EMA_50_SLOPE_ATR'] - df['EMA_50'].shift(slope_period))/(df['ATR'] + 1e-9))
    
    df['BODY_RATIO'] = (abs(df['Close'] - df['Open'])/((df['High'] - df['Low']) + 1e-9))
    df['UPPER_WICK_RATIO'] = ((df['High'] - np.maximum(df['Open'], df['Close']))/((df['High'] - df['Low']) + 1e-9))
    df['LOWER_WICK_RATIO'] = ((np.minimum(df['Open'], df['Close']) - df['Low'])/((df['High'] - df['Low']) + 1e-9))

    df['ATR_EXPANSION'] = (df['ATR']/(df['ATR'].rolling(50).mean() + 1e-9))
    df['ATR_ACCELERATION'] = (df['ATR']/(df['ATR'].shift(5) + 1e-9))
    df['CLOSE_LOCATION'] = ((df['Close'] - df['Low'])/((df['High'] - df['Low']) + 1e-9))
    df['RANGE_EXPANSION'] = ((df['High'] - df['Low'])/((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['DIST_EMA50_ATR'] = ((df['Close'] - df['EMA_50'])/(df['ATR'] + 1e-9))
    df['EMA_200_ATR'] = (df['Close'].ewm(span=200, adjust=False).mean())
    df['DIST_EMA200_ATR'] = ((df['Close'] - df['EMA_200_ATR'])/(df['ATR'] + 1e-9))
    df['TREND_STRENGTH'] = ((df['EMA_50'] - df['EMA_200'])/(df['ATR'] + 1e-9))
    df['BREAKOUT_HIGH_20'] = ((df['Close'] - df['High'].rolling(20).max())/(df['ATR'] + 1e-9))
    df['BREAKOUT_LOW_20'] = ((df['Close'] - df['Low'].rolling(20).min())/(df['ATR'] + 1e-9))
    volume_window = 20
    volume_mean = df['Volume'].rolling(volume_window).mean()
    volume_std = df['Volume'].rolling(volume_window).std()

    df['volume_zscore'] = ((df['Volume'] - volume_mean) / (volume_std + 1e-9))
    short_window = 10
    long_window = 50
    short_range = (df['High'].rolling(short_window).max() -df['Low'].rolling(short_window).min())
    long_range = (df['High'].rolling(long_window).max() -df['Low'].rolling(long_window).min())
    df['rolling_range_ratio'] = short_range / (long_range + 1e-9)
    df['trend_regime'] = ((df['EMA_50'] > df['EMA_200']).astype(int) - (df['EMA_50'] < df['EMA_200']).astype(int))
    df['dist_from_vwap'] = (df['Close'] - df['VWAP']) / df['ATR']
    df['range_expansion'] = df['Range'] / df['Range'].rolling(20).mean()

    def hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    df['hurst100'] = df['Close'].rolling(100).apply(hurst, raw=True)

    df['autocorr_20'] = (df['Close'].rolling(20).corr(df['Close'].shift(1)))
    df['order_flow_imbalance'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
    df['return_1'] = df['Close'].pct_change()
    df['volume_weighted_momentum'] = df['return_1'] * df['Volume']
    df['momentum'] = df['Close'].pct_change(5)
    df['price_acceleration'] = df['momentum'] - df['momentum'].shift(1)
    mid_price = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
    df['liquidity_vacuum'] = (df['Close'] - mid_price) / (df['ATR'] + 1e-9)

    dir_window = 20
    direction = abs(df['Close'] - df['Close'].shift(dir_window))
    volatility = df['Close'].diff().abs().rolling(dir_window).sum()
    df['trend_efficiency'] = direction / volatility

    df[f'rolling_high'] = (df['High'].rolling(25).max().shift(1))
    df[f'rolling_low'] = (df['Low'].rolling(25).min().shift(1))

    df['liquidity_sweep_high'] = ((df['High'] > df['rolling_high']) &
        (df['Close'] < df['rolling_high'])).astype(int)
    
    df['micro_pressure'] = ((df['Close'] - df['Low']) -(df['High'] - df['Close']))
    df['micro_pressure_norm'] = df['micro_pressure'] / df['Range']
    df['volatility_compression'] = df['ATR_14'] / df['ATR_50']

    df['liquidity_sweep_low'] = (
        (df['Low'] < df['rolling_low']) &
        (df['Close'] > df['rolling_low'])).astype(int)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['realized_vol_20'] = (df['log_return'].rolling(20).apply(lambda x: np.sqrt(np.sum(x**2))))

    window = 14
    df['SMA'] = df['Close'].rolling(window).mean()
    df['SMA_slope'] = df['SMA'].diff()
    df['Trend_vs_Range'] = np.where(abs(df['SMA_slope']) > df['Close'].pct_change().rolling(window).std(), 1, 0)
    # 1 = trend, 0 = range

    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window).std()

    df['Hammer'] = np.where(
        (df['High'] - df['Low'] > 3*(df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6), 1, 0)

    df['Shooting_Star'] = np.where(
        (df['High'] - df['Low'] > 3*(df['Open'] - df['Close'])) &
        ((df['High'] - df['Close']) / (0.001 + df['High'] - df['Low']) > 0.6), 1, 0)
    
    df['Doji'] = np.where(
        (abs(df['Close'] - df['Open']) / (0.001 + df['High'] - df['Low']) < 0.1), 1, 0)

    df['Engulfing_Bullish'] = np.where(
        (df['Close'] > df['Open']) & (df['Open'].shift(1) > df['Close'].shift(1)) &
        (df['Close'] > df['Close'].shift(1)) & (df['Open'] < df['Open'].shift(1)), 1, 0)
    df['Engulfing_Bearish'] = np.where(
        (df['Close'] < df['Open']) & (df['Open'].shift(1) < df['Close'].shift(1)) &
        (df['Close'] < df['Close'].shift(1)) & (df['Open'] > df['Open'].shift(1)), 1, 0)
    df['Morning_Star'] = np.where(
        (df['Close'].shift(2) < df['Open'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] > df['Open']) & (df['Open'] > df['Close'].shift(1)) & (df['Open'].shift(1) > df['Close'].shift(2)), 1, 0)
    df['Evening_Star'] = np.where(
        (df['Close'].shift(2) > df['Open'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open']) & (df['Open'] < df['Close'].shift(1)) & (df['Open'].shift(1) < df['Close'].shift(2)), 1, 0)
    df['Three_White_Soldiers'] = np.where(
        (df['Close'] > df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2)), 1, 0)
    df['Three_Black_Crows'] = np.where(
        (df['Close'] < df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2)), 1, 0)
    df['Bullish_Three_Methods'] = np.where(
        (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'] > df['Open'].shift(1)) & (df['Open'].shift(1) > df['Close'].shift(2)), 1, 0)
    df['Bearish_Three_Methods'] = np.where(
        (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'] < df['Open'].shift(1)) & (df['Open'].shift(1) < df['Close'].shift(2)), 1, 0)
    df['Bearish_Evening_Star'] = np.where(
        (df['Close'].shift(2) > df['Open'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open']) & (df['Open'] < df['Close'].shift(1)) & (df['Open'].shift(1) < df['Close'].shift(2)), 1, 0)
    df['Bullish_Three_White_Soldiers'] = np.where(
        (df['Close'] > df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2)), 1, 0)
    df['Bearish_Three_Black_Crows'] = np.where(
        (df['Close'] < df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2)), 1, 0)

    df['AUD_EMA200_DIST'] = ((df['Close'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['AUD_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['AUD_EMA50_SLOPE'] = (df['EMA_50'] - df['EMA_50'].shift(5))
    df['AUD_EMA200_SLOPE'] = (df['EMA_200'] - df['EMA_200'].shift(5))
    df['AUD_TREND_ALIGNMENT'] = (
        (df['EMA_20'] > df['EMA_50']) &(df['EMA_50'] > df['EMA_100']) &
        (df['EMA_100'] > df['EMA_200'])).astype(int)

    df['AUD_ATR_EXPANSION_RATIO'] = (df['ATR_14'] /(df['ATR_50'] + 1e-9))
    df['AUD_ATR_ACCELERATION'] = (df['ATR_14'] -df['ATR_14'].shift(5))
    df['AUD_BB_COMPRESSION_RATIO'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['AUD_BB_EXPANSION_SPEED'] = (df['bollinger_width_20'].diff())
    df['AUD_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /(df['ATR'] + 1e-9))
    df['AUD_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['AUD_CLOSE_LOCATION'] = ((df['Close'] - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['AUD_BODY_RATIO'] = (abs(df['Close'] - df['Open']) /((df['High'] - df['Low']) + 1e-9))
    df['AUD_UPPER_WICK_RATIO'] = ((df['High'] - np.maximum(df['Open'], df['Close'])) /((df['High'] - df['Low']) + 1e-9))
    df['AUD_LOWER_WICK_RATIO'] = ((np.minimum(df['Open'], df['Close']) - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['AUD_RSI_VELOCITY'] = (df['RSI_14'] -df['RSI_14'].shift(3))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = (df['MACD'].ewm(span=9, adjust=False).mean())
    df['MACD_HIST'] = (df['MACD']- df['MACD_SIGNAL'])
    df['MACD_HIST_SLOPE'] = ((df['MACD_HIST'] - df['MACD_HIST'].shift(3))/ (df['ATR'] + 1e-9))
    df['AUD_MACD_HIST_SLOPE'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['AUD_MOMENTUM_PERSISTENCE'] = (
        (df['Close'] > df['Close'].shift(1)) &(df['Close'].shift(1) > df['Close'].shift(2)) &
        (df['Close'].shift(2) > df['Close'].shift(3))).astype(int)

    highest_20 = df['High'].rolling(20).max()
    lowest_20 = df['Low'].rolling(20).min()
    df['AUD_BREAKOUT_STRENGTH'] = ((df['Close'] - highest_20.shift(1)) /(df['ATR'] + 1e-9))
    df['AUD_BREAKDOWN_STRENGTH'] = ((lowest_20.shift(1) - df['Close']) /(df['ATR'] + 1e-9))
    df['AUD_SESSION_RANGE_RATIO'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))

    df['EMA50_SLOPE'] = (df['EMA_50'] - df['EMA_50'].shift(5))
    df['EMA200_SLOPE'] = (df['EMA_200'] - df['EMA_200'].shift(5))
    df['EMA_ALIGNMENT'] = ((df['EMA_20'] > df['EMA_50']) &(df['EMA_50'] > df['EMA_100']) &(df['EMA_100'] > df['EMA_200'])).astype(int)
    df['EMA200_DISTANCE'] = ((df['Close'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['RANGE_EXPANSION'] = ((df['High'] - df['Low']) /(df['ATR'] + 1e-9))
    df['BB_COMPRESSION'] = (df['bollinger_width_20'] /(df['bollinger_width_20'].rolling(20).mean() + 1e-9))
    df['BB_EXPANSION'] = (df['bollinger_width_20'].diff())
    df['RSI_VELOCITY'] = (df['RSI_14'] -df['RSI_14'].shift(3))
    df['RSI_ACCELERATION'] = (df['RSI_VELOCITY'] -df['RSI_VELOCITY'].shift(3))
    df['MACD_SLOPE'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['MOMENTUM_PERSISTENCE'] = ((df['Close'] > df['Close'].shift(1)) &(df['Close'].shift(1) > df['Close'].shift(2)) &(df['Close'].shift(2) > df['Close'].shift(3))).astype(int)

    high20 = df['High'].rolling(20).max()
    low20 = df['Low'].rolling(20).min()
    df['BREAKOUT_STRENGTH'] = ((df['Close'] - high20.shift(1)) /(df['ATR'] + 1e-9))
    df['BREAKDOWN_STRENGTH'] = ((low20.shift(1) - df['Close']) /(df['ATR'] + 1e-9))
    df['HIGH20_DISTANCE'] = ((high20 - df['Close']) /(df['ATR'] + 1e-9))
    df['LOW20_DISTANCE'] = ((df['Close'] - low20) /(df['ATR'] + 1e-9))

    df['AUDJPY_TREND_PERSISTENCE'] = (df['Close'].rolling(10).apply(lambda x: np.sum(np.diff(x) > 0)))
    df['AUDJPY_ATR_BREAKOUT'] = (df['BREAKOUT_STRENGTH'] *df['ATR_EXPANSION'])
    df['AUDJPY_VOLATILITY_SHIFT'] = (df['ATR_14'] /(df['ATR_14'].shift(20) + 1e-9))
    df['AUDJPY_IMPULSE_RATIO'] = (abs(df['Close'] - df['Open']) /((df['High'] - df['Low']) + 1e-9))
    df['AUDJPY_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))

    df['AUDUSD_TREND_STRENGTH'] = df['TREND_STRENGTH']
    df['AUDUSD_ATR_EXPANSION'] = df['ATR_EXPANSION']
    df['AUDUSD_COMPRESSION_RATIO'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['AUDUSD_BREAKOUT_POWER'] = (df['BREAKOUT_STRENGTH'] *df['TREND_STRENGTH'])
    df['AUDUSD_SESSION_RANGE'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(10).mean() + 1e-9))

    df['CADJPY_TREND_PERSISTENCE'] = (df['Close'].rolling(10).apply(lambda x: np.sum(np.diff(x) > 0)))
    df['CADJPY_RANGE_BREAKOUT'] = (df['BREAKOUT_STRENGTH'] *df['RANGE_EXPANSION'])
    df['CADJPY_MOMENTUM_STRENGTH'] = (abs(df['RSI_VELOCITY']) *df['TREND_STRENGTH'])
    df['CADJPY_ATR_ACCELERATION'] = df['ATR_ACCELERATION']
    df['CADJPY_IMPULSE_RATIO'] = (abs(df['Close'] - df['Open']) /((df['High'] - df['Low']) + 1e-9))

    df['EURUSD_MEAN_REVERSION'] = ((df['Close'] - df['EMA_50']) /(df['ATR'] + 1e-9))
    df['EURUSD_VWAP_DISTANCE'] = ((df['Close'] - df['VWAP']) /(df['ATR'] + 1e-9))
    df['EURUSD_RSI_REVERSAL'] = (50 - df['RSI_14'])
    df['EURUSD_RANGE_COMPRESSION'] = ((df['High'] - df['Low']).rolling(10).mean() /((df['High'] - df['Low']).rolling(50).mean() + 1e-9))
    df['EURUSD_LONDON_EXPANSION'] = (df['ATR_14'] /(df['ATR_50'] + 1e-9))

    df['GBPJPY_VOLATILITY_EXPLOSION'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['GBPJPY_BREAKOUT_POWER'] = (df['BREAKOUT_STRENGTH'] *df['VOLATILITY_REGIME'])
    df['GBPJPY_IMPULSE_STRENGTH'] = (abs(df['Close'] - df['Open']) /(df['ATR'] + 1e-9))
    df['GBPJPY_MOMENTUM_ACCELERATION'] = (df['RSI_ACCELERATION'] *df['TREND_STRENGTH'])
    df['GBPJPY_ATR_SURGE'] = (df['ATR_14'] /(df['ATR_14'].shift(10) + 1e-9))

    df['BTC_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['BTC_ATR_EXPLOSION'] = (df['ATR_14'] /(df['ATR_14'].rolling(20).mean() + 1e-9))
    df['BTC_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['BTC_TREND_ACCELERATION'] = (df['EMA_50'].diff(5))
    df['BTC_EMA_DISTANCE'] = ((df['Close'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    high20 = df['High'].rolling(20).max()
    df['BTC_BREAKOUT_PRESSURE'] = ((df['Close'] - high20.shift(1)) /(df['ATR'] + 1e-9))
    df['BTC_COMPRESSION_EXPANSION'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['BTC_BB_EXPANSION_SPEED'] = (df['bollinger_width_20'].diff())
    df['BTC_MOMENTUM_PERSISTENCE'] = ((df['Close'] > df['Close'].shift(1)) &(df['Close'].shift(1) > df['Close'].shift(2)) &(df['Close'].shift(2) > df['Close'].shift(3))).astype(int)
    df['BTC_RSI_VELOCITY'] = (df['RSI_14'] -df['RSI_14'].shift(3))
    df['BTC_RSI_ACCELERATION'] = (df['BTC_RSI_VELOCITY'] -df['BTC_RSI_VELOCITY'].shift(3))
    df['BTC_MACD_ACCELERATION'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['BTC_IMPULSE_STRENGTH'] = (abs(df['Close'] - df['Open']) /(df['ATR'] + 1e-9))
    df['BTC_CLOSE_LOCATION'] = ((df['Close'] - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['BTC_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['BTC_VOL_SHIFT'] = (df['ATR_14'] /(df['ATR_14'].shift(20) + 1e-9))
    df['BTC_TREND_PERSISTENCE'] = (df['Close'].rolling(10).apply(lambda x: np.sum(np.diff(x) > 0)))
    df['BTC_DISTANCE_HIGH20'] = ((high20 - df['Close']) /(df['ATR'] + 1e-9))
    df['BTC_DISTANCE_LOW20'] = ((df['Close'] - low20) /(df['ATR'] + 1e-9))
    df['BTC_BODY_RATIO'] = (abs(df['Close'] - df['Open']) /((df['High'] - df['Low']) + 1e-9))
    df['BTC_UPPER_WICK'] = ((df['High'] - np.maximum(df['Open'], df['Close'])) /((df['High'] - df['Low']) + 1e-9))
    df['BTC_LOWER_WICK'] = ((np.minimum(df['Open'], df['Close']) - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['BTC_TREND_ALIGNMENT'] = ((df['EMA_20'] > df['EMA_50']) &
        (df['EMA_50'] > df['EMA_100']) &(df['EMA_100'] > df['EMA_200'])).astype(int)
    df['BTC_BREAKOUT_VOLATILITY'] = (df['BTC_BREAKOUT_PRESSURE'] *df['BTC_VOLATILITY_REGIME'])

    df['GBPUSD_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['GBPUSD_TREND_ACCELERATION'] = (df['EMA_50'] - df['EMA_50'].shift(5))
    df['GBPUSD_ATR_EXPLOSION'] = (df['ATR_14'] /(df['ATR_14'].rolling(20).mean() + 1e-9))
    high20 = df['High'].rolling(20).max()
    df['GBPUSD_BREAKOUT_POWER'] = ((df['Close'] - high20.shift(1)) /(df['ATR'] + 1e-9))
    df['GBPUSD_MOMENTUM_PERSISTENCE'] = ((df['Close'] > df['Close'].shift(1)) &
    (df['Close'].shift(1) > df['Close'].shift(2)) &(df['Close'].shift(2) > df['Close'].shift(3))).astype(int)
    df['GBPUSD_RSI_VELOCITY'] = (df['RSI_14'] - df['RSI_14'].shift(3))
    df['GBPUSD_RSI_ACCELERATION'] = (df['GBPUSD_RSI_VELOCITY'] -df['GBPUSD_RSI_VELOCITY'].shift(3))
    df['GBPUSD_MACD_ACCELERATION'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['GBPUSD_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['GBPUSD_BREAKOUT_VOLATILITY'] = (df['GBPUSD_BREAKOUT_POWER'] *(df['ATR_14']/(df['ATR_100']+1e-9)))
    df['NZDCAD_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['NZDCAD_ATR_EXPANSION'] = (df['ATR_14'] /(df['ATR_50'] + 1e-9))
    df['NZDCAD_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['NZDCAD_BREAKOUT_STRENGTH'] = ((df['Close'] - df['High'].rolling(20).max().shift(1)) /(df['ATR'] + 1e-9))
    df['NZDCAD_BREAKDOWN_STRENGTH'] = ((df['Low'].rolling(20).min().shift(1) - df['Close']) /(df['ATR'] + 1e-9))
    df['NZDCAD_COMPRESSION_RATIO'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['NZDCAD_RSI_VELOCITY'] = (df['RSI_14'] - df['RSI_14'].shift(3))
    df['NZDCAD_MACD_SLOPE'] = (df['MACD_HIST'] - df['MACD_HIST'].shift(3))
    df['NZDCAD_IMPULSE_RATIO'] = (abs(df['Close'] - df['Open']) /((df['High'] - df['Low']) + 1e-9))
    df['NZDCAD_CLOSE_LOCATION'] = ((df['Close'] - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['NZDUSD_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['NZDUSD_ATR_EXPANSION'] = (df['ATR_14'] /(df['ATR_50'] + 1e-9))
    df['NZDUSD_TREND_ALIGNMENT'] = ((df['EMA_20'] > df['EMA_50']) &(df['EMA_50'] > df['EMA_100']) &(df['EMA_100'] > df['EMA_200'])).astype(int)
    df['NZDUSD_BREAKOUT_POWER'] = ((df['Close'] - df['High'].rolling(20).max().shift(1)) /(df['ATR'] + 1e-9))
    df['NZDUSD_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['NZDUSD_RSI_VELOCITY'] = (df['RSI_14'] - df['RSI_14'].shift(3))
    df['NZDUSD_RSI_ACCELERATION'] = (df['NZDUSD_RSI_VELOCITY'] -df['NZDUSD_RSI_VELOCITY'].shift(3))
    df['NZDUSD_MACD_ACCELERATION'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['NZDUSD_COMPRESSION_RATIO'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['NZDUSD_SESSION_RANGE'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(10).mean() + 1e-9))

    df['XAU_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['XAU_ATR_EXPLOSION'] = (df['ATR_14'] /(df['ATR_14'].rolling(20).mean() + 1e-9))
    df['XAU_ATR_ACCELERATION'] = (df['ATR_14'] -df['ATR_14'].shift(5))
    df['XAU_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['XAU_TREND_ACCELERATION'] = (df['EMA_50'] -df['EMA_50'].shift(5))
    df['XAU_EMA_DISTANCE'] = ((df['Close'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['XAU_TREND_ALIGNMENT'] = ((df['EMA_20'] > df['EMA_50']) &(df['EMA_50'] > df['EMA_100']) &(df['EMA_100'] > df['EMA_200'])).astype(int)
    high20 = df['High'].rolling(20).max()
    df['XAU_BREAKOUT_PRESSURE'] = ((df['Close'] - high20.shift(1)) /(df['ATR'] + 1e-9))
    low20 = df['Low'].rolling(20).min()
    df['XAU_BREAKDOWN_PRESSURE'] = ((low20.shift(1) - df['Close']) /(df['ATR'] + 1e-9))
    df['XAU_BREAKOUT_VOLATILITY'] = (df['XAU_BREAKOUT_PRESSURE'] *df['XAU_VOLATILITY_REGIME'])
    df['XAU_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['XAU_COMPRESSION_RATIO'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['XAU_BB_EXPANSION_SPEED'] = (df['bollinger_width_20'].diff())
    df['XAU_RSI_VELOCITY'] = (df['RSI_14'] -df['RSI_14'].shift(3))
    df['XAU_RSI_ACCELERATION'] = (df['XAU_RSI_VELOCITY'] -df['XAU_RSI_VELOCITY'].shift(3))
    df['XAU_MACD_ACCELERATION'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['XAU_MOMENTUM_PERSISTENCE'] = ((df['Close'] > df['Close'].shift(1)) &(df['Close'].shift(1) > df['Close'].shift(2)) &(df['Close'].shift(2) > df['Close'].shift(3))).astype(int)
    df['XAU_IMPULSE_STRENGTH'] = (abs(df['Close'] - df['Open']) /(df['ATR'] + 1e-9))
    df['XAU_CLOSE_LOCATION'] = ((df['Close'] - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['XAU_UPPER_WICK'] = ((df['High'] - np.maximum(df['Open'], df['Close'])) /((df['High'] - df['Low']) + 1e-9))
    df['XAU_LOWER_WICK'] = ((np.minimum(df['Open'], df['Close']) - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['XAU_LIQUIDITY_SWEEP_HIGH'] = ((df['High'] > high20.shift(1)) &(df['Close'] < high20.shift(1))).astype(int)
    df['XAU_LIQUIDITY_SWEEP_LOW'] = ((df['Low'] < low20.shift(1)) &(df['Close'] > low20.shift(1))).astype(int)
    df['XAU_DISTANCE_HIGH20'] = ((high20 - df['Close']) /(df['ATR'] + 1e-9))
    df['XAU_DISTANCE_LOW20'] = ((df['Close'] - low20) /(df['ATR'] + 1e-9))

    df['XAG_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['XAG_ATR_EXPLOSION'] = (df['ATR_14'] /(df['ATR_14'].rolling(20).mean() + 1e-9))
    df['XAG_ATR_ACCELERATION'] = (df['ATR_14'] - df['ATR_14'].shift(5))
    df['XAG_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['XAG_TREND_ACCELERATION'] = (df['EMA_50'] -df['EMA_50'].shift(5))
    df['XAG_EMA_DISTANCE'] = ((df['Close'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['XAG_TREND_ALIGNMENT'] = ((df['EMA_20'] > df['EMA_50']) &(df['EMA_50'] > df['EMA_100']) &
    (df['EMA_100'] > df['EMA_200'])).astype(int)
    high20 = df['High'].rolling(20).max()
    df['XAG_BREAKOUT_PRESSURE'] = ((df['Close'] - high20.shift(1)) /(df['ATR'] + 1e-9))
    low20 = df['Low'].rolling(20).min()
    df['XAG_BREAKDOWN_PRESSURE'] = ((low20.shift(1) - df['Close']) /(df['ATR'] + 1e-9))
    df['XAG_BREAKOUT_VOLATILITY'] = (df['XAG_BREAKOUT_PRESSURE'] *df['XAG_VOLATILITY_REGIME'])
    df['XAG_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['XAG_COMPRESSION_RATIO'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['XAG_BB_EXPANSION_SPEED'] = (df['bollinger_width_20'].diff())
    df['XAG_RSI_VELOCITY'] = (df['RSI_14'] -df['RSI_14'].shift(3))
    df['XAG_RSI_ACCELERATION'] = (df['XAG_RSI_VELOCITY'] -df['XAG_RSI_VELOCITY'].shift(3))
    df['XAG_MACD_ACCELERATION'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['XAG_MOMENTUM_PERSISTENCE'] = ((df['Close'] > df['Close'].shift(1)) &(df['Close'].shift(1) > df['Close'].shift(2)) &
    (df['Close'].shift(2) > df['Close'].shift(3))).astype(int)
    df['XAG_IMPULSE_STRENGTH'] = (abs(df['Close'] - df['Open']) /(df['ATR'] + 1e-9))
    df['XAG_CLOSE_LOCATION'] = ((df['Close'] - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['XAG_UPPER_WICK'] = ((df['High'] - np.maximum(df['Open'], df['Close'])) /((df['High'] - df['Low']) + 1e-9))
    df['XAG_LOWER_WICK'] = ((np.minimum(df['Open'], df['Close']) - df['Low']) /((df['High'] - df['Low']) + 1e-9))
    df['XAG_LIQUIDITY_SWEEP_HIGH'] = ((df['High'] > high20.shift(1)) &(df['Close'] < high20.shift(1))).astype(int)
    df['XAG_LIQUIDITY_SWEEP_LOW'] = ((df['Low'] < low20.shift(1)) &(df['Close'] > low20.shift(1))).astype(int)
    df['XAG_DISTANCE_HIGH20'] = ((high20 - df['Close']) /(df['ATR'] + 1e-9))
    df['XAG_DISTANCE_LOW20'] = ((df['Close'] - low20) /(df['ATR'] + 1e-9))

    df['USDJPY_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['USDJPY_EMA50_SLOPE'] = (df['EMA_50'] -df['EMA_50'].shift(5))
    df['USDJPY_EMA200_SLOPE'] = (df['EMA_200'] -df['EMA_200'].shift(10))
    df['USDJPY_TREND_ALIGNMENT'] = ((df['EMA_20'] > df['EMA_50']) &(df['EMA_50'] > df['EMA_200'])).astype(int)
    df['USDJPY_ATR_EXPANSION'] = (df['ATR_14'] /(df['ATR_14'].rolling(50).mean() + 1e-9))
    high20 = df['High'].rolling(20).max()
    df['USDJPY_BREAKOUT_STRENGTH'] = ((df['Close'] - high20.shift(1)) /(df['ATR'] + 1e-9))
    low20 = df['Low'].rolling(20).min()
    df['USDJPY_BREAKDOWN_STRENGTH'] = ((low20.shift(1) - df['Close']) /(df['ATR'] + 1e-9))
    df['USDJPY_MOMENTUM_PERSISTENCE'] = ((df['Close'] > df['Close'].shift(1)) &(df['Close'].shift(1) > df['Close'].shift(2)) &
    (df['Close'].shift(2) > df['Close'].shift(3))).astype(int)
    df['USDJPY_RSI_VELOCITY'] = (df['RSI_14'] -df['RSI_14'].shift(3))
    df['USDJPY_RSI_ACCELERATION'] = (df['USDJPY_RSI_VELOCITY'] -df['USDJPY_RSI_VELOCITY'].shift(3))
    df['USDJPY_MACD_SLOPE'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(3))
    df['USDJPY_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['USDJPY_EMA200_DISTANCE'] = ((df['Close'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['USDJPY_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['USDJPY_IMPULSE_RATIO'] = (abs(df['Close'] - df['Open']) /(df['ATR'] + 1e-9))

    df['USDSEK_VOLATILITY_REGIME'] = (df['ATR_14'] /(df['ATR_100'] + 1e-9))
    df['USDSEK_ATR_EXPLOSION'] = (df['ATR_14'] /(df['ATR_14'].rolling(20).mean() + 1e-9))
    df['USDSEK_ATR_ACCELERATION'] = (df['ATR_14'] -df['ATR_14'].shift(5))
    df['USDSEK_TREND_STRENGTH'] = (abs(df['EMA_50'] - df['EMA_200']) /(df['ATR'] + 1e-9))
    df['USDSEK_REVERSAL_PRESSURE'] = ((df['Close'] - df['EMA_50']) /(df['ATR'] + 1e-9))
    df['USDSEK_BB_COMPRESSION'] = (df['bollinger_width_20'] /(df['ATR'] + 1e-9))
    df['USDSEK_BB_EXPANSION'] = (df['bollinger_width_20'].diff())
    df['USDSEK_RANGE_EXPANSION'] = ((df['High'] - df['Low']) /((df['High'] - df['Low']).rolling(20).mean() + 1e-9))
    df['USDSEK_RSI_REVERSAL'] = (df['RSI_14'] -df['RSI_14'].shift(5))
    df['USDSEK_MACD_ACCELERATION'] = (df['MACD_HIST'] -df['MACD_HIST'].shift(5))
    high20 = df['High'].rolling(20).max()
    df['USDSEK_SWEEP_HIGH'] = ((df['High'] > high20.shift(1)) &(df['Close'] < high20.shift(1))).astype(int)
    low20 = df['Low'].rolling(20).min()
    df['USDSEK_SWEEP_LOW'] = ((df['Low'] < low20.shift(1)) &(df['Close'] > low20.shift(1))).astype(int)
    df['USDSEK_DISTANCE_HIGH20'] = ((high20 - df['Close']) /(df['ATR'] + 1e-9))
    df['USDSEK_DISTANCE_LOW20'] = ((df['Close'] - low20) /(df['ATR'] + 1e-9))
    df['USDSEK_IMPULSE_STRENGTH'] = (abs(df['Close'] - df['Open']) /(df['ATR'] + 1e-9))


    threshold = 0.7
    nan_fraction = df.isna().mean()
    cols_to_keep = nan_fraction[nan_fraction < threshold].index
    df = df[cols_to_keep]
    # print("Columns kept:", len(cols_to_keep))
    # print("Columns dropped:", len(df.columns) - len(cols_to_keep))

    df.set_index("Date", inplace=True)
    return df



def normalize_symbol(symbol):
    try:
        return re.match(r"[A-Z]+",symbol).group()
    except:
        return symbol


def rollover_sleep(mt5, symbol):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return

    # ==========================================
    # GET MT5 SERVER TIME
    # ==========================================
    print('tick.time :',tick.time)

    server_time = datetime.fromtimestamp(tick.time)
    print(server_time)

    LOGGER.info(f"MT5 SERVER TIME: {server_time}")

    hour = server_time.hour

    # ==========================================
    # CHECK ROLLOVER WINDOW
    # ==========================================

    if hour >= 23 or hour < 3:

        LOGGER.info("ROLL OVER PERIOD DETECTED")

        # ======================================
        # CREATE TARGET WAKE TIME
        # ======================================

        wake_time = server_time.replace(
            hour=3,
            minute=10,
            second=40,
            microsecond=0
        )

        # ======================================
        # HANDLE NEXT DAY CASE
        # ======================================

        if hour >= 23:

            wake_time += timedelta(days=1)

        # ======================================
        # CALCULATE SLEEP SECONDS
        # ======================================

        sleep_seconds = (wake_time - server_time).total_seconds()

        if sleep_seconds > 0:

            LOGGER.info(f"SLEEPING FOR {sleep_seconds:.0f} SECONDS")

            time.sleep(sleep_seconds)

        LOGGER.info("ROLLOVER FINISHED")



def wait_for_new_candle(mt5, symbol, timeframe):

    LOGGER.info(f"\n [WAITING FOR NEW CANDLE : {symbol}]\n")

    # ==========================================
    # STAGE 1
    # Efficient local waiting
    # ==========================================

    while True:

        now = datetime.now()

        minute = now.minute
        second = now.second

        # Seconds until next 5m candle
        seconds_remaining = ((4 - (minute % 5)) * 60) + (60 - second)

        # Fix exact boundary
        if seconds_remaining == 300:
            seconds_remaining = 0

        # Start precise checking near candle
        if seconds_remaining <= 10:
            LOGGER.info("\n[STARTING PRECISE MT5 CHECK]\n")
            break

        sleep_time = max(seconds_remaining - 5, 1)

        LOGGER.info(
            f"Waiting efficiently... "
            f"{seconds_remaining}s remaining | "
            f"Sleeping {sleep_time}s"
        )

        time.sleep(sleep_time)

    # ==========================================
    # STAGE 2
    # Precise MT5 synchronization
    # ==========================================

    last_candle = mt5.copy_rates_from_pos(symbol,timeframe,0,1)

    if last_candle is None or len(last_candle) == 0:
        raise RuntimeError("Failed to fetch MT5 candle")

    last_time = last_candle[0]['time']

    while True:

        current_candle = mt5.copy_rates_from_pos(symbol,timeframe,0,1)

        if current_candle is None or len(current_candle) == 0:
            time.sleep(0.5)
            continue

        current_time = current_candle[0]['time']

        if current_time != last_time:

            LOGGER.info(f"\n >>>> NEW CANDLE DETECTED : {symbol} <<<<\n")

            return current_time

        time.sleep(0.3)


def forex_market_open(mt5, symbol):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return False

    server_time = datetime.fromtimestamp(tick.time)

    weekday = server_time.weekday()
    hour = server_time.hour

    # Saturday
    if weekday == 5:
        return False

    # Sunday before open
    if weekday == 6 and hour < 22:
        return False

    return True



def Data_Prediction(mt5,SYMBOL,HIGH_TARGETS,HELPER_MODELS,ext_func = False):
    
    LOGGER.info(f"\n[DATA PREDICTION ON NEW CANDLE : {SYMBOL}]\n")

    TIMEFRAME = mt5.TIMEFRAME_M10
    N_BARS = 600

    if ext_func:
        rates = mt5.copy_rates_from_pos(SYMBOL,TIMEFRAME,0,N_BARS )
    else:
        wait_for_new_candle(mt5,SYMBOL, TIMEFRAME)
        rates = mt5.copy_rates_from_pos(SYMBOL,TIMEFRAME,1,N_BARS )

    if rates is None or len(rates) < N_BARS:
        mt5.shutdown()
        raise RuntimeError("[[BAD]] Failed to fetch enough closed candles")

    data = pd.DataFrame(rates)
    data['Date'] = pd.to_datetime(data['time'], unit='s')

    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume',
        'spread': 'Spread'
    }, inplace=True)

    new_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']]
    new_df.sort_values('Date', inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    LOGGER.info(new_df.tail())

    df = apply_features(new_df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    up_moves = {}
    down_moves = {}
    for target in HIGH_TARGETS:

        LOGGER.info(f'\n [[ CURRENTLY PREDICTING TARGET : {target} ]]')

        help_model = HELPER_MODELS[target]["model"]
        help_cols = HELPER_MODELS[target]["features"]
        
        current_candle = df.tail(1)

        X_help = current_candle[help_cols]
        # Safety check
        assert set(help_cols) == set(X_help.columns)

        help_proba = help_model.predict_proba(X_help)[0]

        up_prob = help_proba[1]
        down_prob = help_proba[0]

        LOGGER.info(f'HELPER UP PROB : {help_proba[1]}')
        LOGGER.info(f'HELPER DOWN PROB : {help_proba[0]}')

        up_moves[target] = round(up_prob, 2)
        down_moves[target] = round(down_prob, 2)

        direction = "UP" if up_prob > down_prob else "DOWN"
        LOGGER.info(f"HELP MODEL PROBABILITY : {help_proba}")
        LOGGER.info(f">>> FINAL → {direction} ({round(max(up_prob, down_prob)*100,2)}%)")

    return up_moves, down_moves , df







def calc_lot_size(mt5, balance, risk_percent, sl_pips, pip_value_per_lot, SYMBOL):

    info = mt5.symbol_info(SYMBOL)

    min_lot = info.volume_min
    max_lot = info.volume_max
    step = info.volume_step

    risk_amount = balance * (risk_percent / 100)
    lot_cal = risk_amount / (sl_pips * pip_value_per_lot)

    lot = max(min_lot, min(lot_cal, max_lot))
    lot = np.floor(lot / step) * step

    return lot



def normalize_lot(lot, vol_min, vol_max, vol_step):
    lot = max(vol_min, min(lot, vol_max))
    lot = np.floor(lot / vol_step) * vol_step
    return round(lot, 2)



def split_pair(symbol):

    base = symbol[:3]
    quote = symbol[3:]

    return base, quote


def get_trade_bias(symbol, trade_type):

    """
    Returns currency exposure.

    BUY EURUSD:
        EUR = +1
        USD = -1

    SELL EURUSD:
        EUR = -1
        USD = +1
    """

    base, quote = split_pair(symbol)

    if trade_type == "BUY":

        return {
            base: 1,
            quote: -1
        }

    else:

        return {
            base: -1,
            quote: 1
        }


def correlation_check(mt5,new_symbol, new_trade_type):

    positions = mt5.positions_get()

    if positions is None:
        return True

    new_bias = get_trade_bias(new_symbol, new_trade_type)

    total_exposure = {}

    # ==========================================
    # EXISTING OPEN TRADES
    # ==========================================

    for pos in positions:

        symbol = pos.symbol

        if pos.type == mt5.ORDER_TYPE_BUY:
            trade_type = "BUY"

        else:
            trade_type = "SELL"

        bias = get_trade_bias(symbol, trade_type)

        for currency, value in bias.items():

            if currency not in total_exposure:
                total_exposure[currency] = 0

            total_exposure[currency] += value

    # ==========================================
    # CHECK NEW TRADE IMPACT
    # ==========================================

    for currency, value in new_bias.items():

        current = total_exposure.get(currency, 0)

        new_total = current + value

        # ======================================
        # OVEREXPOSURE LIMIT
        # ======================================

        if abs(new_total) > 2:

            LOGGER.info(f"BLOCKED: Too much exposure on {currency}")

            return False

    return True



def spread_filter(mt5, SYMBOL, SL_distance):

    tick = mt5.symbol_info_tick(SYMBOL)

    if tick is None:
        LOGGER.info("[[BAD]] No tick data")
        return False
    
    ask_price = tick.ask
    bid_price = tick.bid

    spread = ask_price - bid_price
    spread_ratio = spread / SL_distance

    symbol_info = mt5.symbol_info(SYMBOL)
    spread_points = spread / symbol_info.point
    
    sl_pips = SL_distance / symbol_info.point

    LOGGER.info(f"""
    ====================================================================
    [[ ASK VALUE]] :{ask_price}
    [[ BID VALUE]] :{bid_price}
    [[WARNING]]>>>>>> ACTUAL SPREAD VALUE IS {spread} <<<<<<<<<<<<<<     
    Spread Ratio: {spread_ratio:.2%}    
    [[WARNING]] SPREAD POINT VALUE IS { round(spread_points,2) } PIPS   
    SL DISTANCE   : {sl_pips:.2f} pips
    ====================================================================
    """)

    min_stop_distance = symbol_info.point * 10
    if SL_distance < min_stop_distance:
        LOGGER.info("[[BAD]] SL too small, skipping trade")
        return False

    if spread_ratio > 0.15:

        LOGGER.info(
            f"[[WARNING]] Spread Too Expensive Relative To SL | "
            f"Spread Ratio: {spread_ratio:.2%}"
        )

        return False
    LOGGER.info("[[GOOD]]  SPREAD PIP SIZING CONFIRMATION PASSED ")
    return True



def check_trade_result(mt5, result):
    if result is None:
        LOGGER.info("[[BAD]] Order failed: result is None")
        LOGGER.info(f"MT5 last error: {mt5.last_error()}" )
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        LOGGER.info("[[BAD]] Order rejected")
        LOGGER.info(f"Retcode: {result.retcode}" )
        LOGGER.info(f"Comment: {result.comment}" )
        LOGGER.info(f"Request ID: {result.request_id}" )
        return False

    LOGGER.info("[[GOOD]] Trade placed successfully")
    LOGGER.info(f"Order Ticket: {result.order}" )
    LOGGER.info(f"Deal Ticket: {result.deal}" )
    LOGGER.info(f"Volume: {result.volume}" )
    LOGGER.info(f"Price: {result.price}" )
    return True

def get_symbol_volume_info(mt5, symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError("Failed to get symbol info")

    return {
        "min": info.volume_min,
        "max": info.volume_max,
        "step": info.volume_step
    }


def place_sell(mt5,symbol, lot, entry_price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 25,
        "magic": 10002,
        "comment": "Auto SELL",
        "type_time": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)
    check_trade_result(mt5, result)
    return result



def place_buy(mt5,symbol, lot, entry_price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 25,
        "magic": 10001,
        "comment": "Auto BUY",
        "type_time": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)
    check_trade_result(mt5, result)
    return result




def drop_duplicate(path):
    all_df = pd.read_csv(path)
    all_df = all_df.drop_duplicates(keep='first')
    all_df = all_df.reset_index()
    all_df.drop(['index'], axis=1, inplace=True)
    all_df.to_csv(path, index=False)


def create_targets(df):
    horizons = {
        "T_5M": 1,
        "T_10M": 2,
        "T_15M": 3,
        "T_20M": 4,
        "T_30M": 6
    }
    for name, step in horizons.items():
        future_close = df['Close'].shift(-step)
        log_return = np.log(future_close / df['Close'])
        df[name] = (log_return > 0).astype(int)
    df = df.iloc[:-6]
    return df


def create_hl_targets(df):
    horizons = {
        "THL_5M": 1,
        "THL_10M": 2,
        "THL_15M": 3,
        "THL_20M": 4,
        "THL_30M": 6
    }
    for name, step in horizons.items():
        df[name] = df['High'].shift(-step)
    return df


def trade_backtest(df, model, feature_cols, threshold=55, atr_sl=1.5, atr_tp=4.5, spread_pips=1.2, slippage_pips=0.2, pip_value=0.0001):
    trades = []
    spread = spread_pips * pip_value
    slippage = slippage_pips * pip_value
    i = 0 

    while i < len(df) - 1:
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        X = row[feature_cols].values.reshape(1, -1)
        proba = model.predict_proba(X)[0]
        up_conf, down_conf = proba[1] * 100, proba[0] * 100

        if max(up_conf, down_conf) < threshold:
            i += 1
            continue

        direction = "BUY" if up_conf > down_conf else "SELL"
        atr = row["ATR"]

        if direction == "BUY":
            entry = next_row["Open"] + spread + slippage
            sl = entry - (atr_sl * atr)
            tp = entry + (atr_tp * atr)
        else:
            entry = next_row["Open"] - slippage
            # Sell exit triggers at Ask price (Bid + Spread)
            sl = entry + (atr_sl * atr)
            tp = entry - (atr_tp * atr)

        for j in range(i + 1, len(df)):
            candle = df.iloc[j]
            
            if direction == "BUY":
                # Close price for BUY is Bid (Standard df prices)
                if candle["Low"] <= sl and candle["High"] >= tp:
                    trades.append(("LOSS", direction, i, j)) # Conservative
                    i = j; break
                elif candle["Low"] <= sl:
                    trades.append(("LOSS", direction, i, j)); i = j; break
                elif candle["High"] >= tp:
                    trades.append(("WIN", direction, i, j)); i = j; break
            else:
                # Close price for SELL is Ask (Bid + Spread)
                candle_high_ask = candle["High"] + spread
                candle_low_ask = candle["Low"] + spread
                
                if candle_high_ask >= sl and candle_low_ask <= tp:
                    trades.append(("LOSS", direction, i, j)) # Conservative
                    i = j; break
                elif candle_high_ask >= sl:
                    trades.append(("LOSS", direction, i, j)); i = j; break
                elif candle_low_ask <= tp:
                    trades.append(("WIN", direction, i, j)); i = j; break
        else:
            i += 1
    return trades



def analyze_results(trades):
    total = len(trades)
    wins = sum(1 for t in trades if t[0] == "WIN")
    losses = total - wins
    win_rate = round((wins / total) * 100, 2) if total > 0 else 0

    LOGGER.info("Total Trades:", total)
    LOGGER.info("Wins:", wins)
    LOGGER.info("Losses:", losses)
    LOGGER.info("Win Rate:", win_rate, "%")
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate
    }




def get_pip_info(mt5, symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info not found for {symbol}")

    tick_size = info.trade_tick_size
    tick_value = info.trade_tick_value
    digits = info.digits

    # Determine pip size
    if digits in (3, 5):
        pip_size = tick_size * 10
    else:
        pip_size = tick_size

    # Pip value per 1 lot
    pip_value_per_lot = (pip_size / tick_size) * tick_value

    return {
        "pip_size": pip_size,
        "pip_value_per_lot": pip_value_per_lot,
        "tick_size": tick_size,
        "tick_value": tick_value
    }



TRADE_LOG_FILE = "CSV_FILES/Trade_log.csv"

def log_trade(mt5,symbol,direction,entry_price,SL,TP,lot_size,proba_up,proba_down,order_result):

    account_info = mt5.account_info()
    current_balance = account_info.balance

    # Load existing log if it exists
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        prev_balance = df.iloc[-1]["Balance"] if not df.empty else current_balance
    else:
        df = pd.DataFrame()
        prev_balance = current_balance

    # Determine PnL status
    if current_balance > prev_balance:
        pnl_status = "Profit"
    elif current_balance < prev_balance:
        pnl_status = "Loss"
    else:
        pnl_status = "No Change"

    # Create new row
    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Direction": direction,
        "Entry": entry_price,
        "SL": SL,
        "TP": TP,
        "Lot": lot_size,
        "Proba_UP": proba_up,
        "Proba_DOWN": proba_down,
        "OrderResult": order_result,
        "Balance": current_balance,
        "PnL_Status": pnl_status
    }

    # Append row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back to CSV
    df.to_csv(TRADE_LOG_FILE, index=False)

    LOGGER.info(f"[[GOOD]] Trade logged. Current Balance: {current_balance}, Status: {pnl_status}")


def saving_files(data,path):
    df = pd.DataFrame(data)
    LOGGER.info(df.to_string())

    try:
        df2 = pd.read_csv(path)
        all_df = pd.concat([df2, df], ignore_index=True)
        all_df.to_csv(path, index=False)
        LOGGER.info(' ------------------------------------ ALL FILES SAVED  ------------------------------------- \n \n')

    except:
        df.to_csv(path, index=False)
        LOGGER.info('============================= SECOND FILE SAVED ==========================')


def get_monday(date_obj):
    return date_obj - timedelta(days=date_obj.weekday())

def ff_week_format(date_obj):
    return date_obj.strftime("%b").lower() + str(date_obj.day) + "." + str(date_obj.year)

def ff_day_format(date_obj):
    return date_obj.strftime("%b").lower() + str(date_obj.day) + "." + str(date_obj.year)


def normalize_datetime(date_str, time_str, year):
    if not date_str or not time_str:
        return None

    if time_str.lower() == "all day":
        time_str = "12:00am"

    dt_str = f"{date_str} {year} {time_str}"

    try:
        dt = datetime.strptime(dt_str, "%a %b %d %Y %I:%M%p")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None



def confirmation_entry(df, direction):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == "BUY":
        return (
            last["Close"] > last["Open"] and
            last["Close"] > prev["Close"]
        )

    elif direction == "SELL":
        return (
            last["Close"] < last["Open"] and
            last["Close"] < prev["Close"]
        )

    return False


def hybrid_entry(mt5, SYMBOL, df, direction, atr_value, pullback_ratio=0.15, timeout=5):
    """
    Hybrid Entry:
    1. Confirm direction
    2. Wait small pullback
    """

    # -------------------------
    # STEP 1: CONFIRMATION
    # -------------------------
    if not confirmation_entry(df, direction):
        LOGGER.info("[[LOADING]] No confirmation yet")
        return None

    LOGGER.info(f"[[GOOD]] Confirmation passed FOR {direction} SIGNAL")

    tick = mt5.symbol_info_tick(SYMBOL)

    if direction == "BUY":
        initial_price = tick.ask
        pullback_level = initial_price - (pullback_ratio * atr_value)

    else:
        initial_price = tick.bid
        pullback_level = initial_price + (pullback_ratio * atr_value)

    LOGGER.info(f"[[TARGET]] Waiting for pullback @ {pullback_level}")

    # -------------------------
    # STEP 2: MICRO PULLBACK
    # -------------------------
    for _ in range(timeout):
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if direction == "BUY" else tick.bid

        # BUY pullback
        if direction == "BUY" and price <= pullback_level:
            LOGGER.info("[[GOOD]] Pullback hit (BUY)")
            return price

        # SELL pullback
        if direction == "SELL" and price >= pullback_level:
            LOGGER.info("[[GOOD]] Pullback hit (SELL)")
            return price

        time.sleep(1)

    # -------------------------
    # FALLBACK: ENTER MARKET
    # -------------------------
    LOGGER.info("[[WARNING]] No pullback → entering at market")
    tick = mt5.symbol_info_tick(SYMBOL)
    return tick.ask if direction == "BUY" else tick.bid






def Entry_Filtering(mt5, SYMBOL, df, direction, atr_value):

    tick = mt5.symbol_info_tick(SYMBOL)

    last = df.iloc[-1]
    prev = df.iloc[-2]
        
    # =========================
    # 2. MOMENTUM STRENGTH
    # =========================
    body_size = abs(last["Close"] - last["Open"])
    if body_size < 0.10 * atr_value:
        LOGGER.info("[[BAD]] Weak momentum candle")
        return None

    # Confirmation (relaxed)
    if direction == "BUY":
        if last["Close"] <= prev["Close"]:
            return None

    elif direction == "SELL":
        if last["Close"] >= prev["Close"]:
            return None

    LOGGER.info("[[GOOD]] Confirmation passed")

    # =========================
    # 4. STRUCTURED PULLBACK
    # =========================
    pullback_ratio = 0.3
    invalid_ratio = 0.8   
    max_wait = 60*3  # 60*3 = 3 minutes

    # =========================
    # 5. WAIT FOR PULLBACK (WITH INVALIDATION)
    # ========================= 

    entry_base = tick.ask if direction == "BUY" else tick.bid

    pullback_level = entry_base - (pullback_ratio * atr_value) if direction == "BUY" else entry_base + (pullback_ratio * atr_value)

    invalidation_level = entry_base - (invalid_ratio * atr_value) if direction == "BUY" else entry_base + (invalid_ratio * atr_value)
    LOGGER.info(f"[[TARGET]] Waiting pullback @ {pullback_level}")

    pullback_hit = False
    lowest_price = entry_base

    for _ in range(max_wait):

        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if direction == "BUY" else tick.bid

        # =========================
        # 1. INVALIDATION
        # =========================
        if direction == "BUY" and price <= invalidation_level:
            LOGGER.info('BUY : PRICE HIT INVALIDATION POINT, SO TRADE IS ABANDONED')
            return None

        if direction == "SELL" and price >= invalidation_level:
            LOGGER.info('SELL : PRICE HIT INVALIDATION POINT, SO TRADE IS ABANDONED')
            return None

        # =========================
        # 2. TRACK PULLBACK
        # =========================
        if direction == "BUY":
            if price < lowest_price:
                lowest_price = price

            # pullback reached
            if price <= pullback_level:
                pullback_hit = True

            # WAIT for reversal after pullback
            if pullback_hit and price > lowest_price + (0.22 * atr_value):
                LOGGER.info("[[GOOD]] Confirmed pullback entry (BUY)")
                return price

        else:
            if price > lowest_price:
                lowest_price = price

            if price >= pullback_level:
                pullback_hit = True

            if pullback_hit and price < lowest_price - (0.22 * atr_value):
                LOGGER.info("[[GOOD]] Confirmed pullback entry (SELL)")
                return price

        # =========================
        # 3. MOMENTUM ENTRY
        # =========================
        if direction == "BUY" and price >= entry_base + (0.18 * atr_value):
            LOGGER.info("[[GOOD]] Confirmed Clean movement entry (BUY)")
            return price

        if direction == "SELL" and price <= entry_base - (0.18 * atr_value):
            LOGGER.info("[[GOOD]] Confirmed Clean movement entry (SELL)")
            return price

        time.sleep(1)

    # =========================
    # 6. NO TRADE IF NO PULLBACK
    # =========================
    LOGGER.info("[[WARNING]] No clean pullback → skip trade")
    return None



def normalize_volume(volume, min_lot, step):
    return round(max(min_lot, round(volume / step) * step), 2)


TRADE_STATE = {}  # ticket -> state
def move_sl_and_partial_close(mt5, SYMBOL, atr_value,HIGH_TARGETS,HELPER_MODELS,W_threshold):

    PARTIAL_CLOSE_PCT = 0.4 # AMOUT OF LOT SIZE THAT CAN BE CLOSED E.G 0.4 = 40% OF LOT SIZE
    PARTIAL_CLOSE_LEVEL = 0.35 # LEVEL THAT PARTIAL PROFIT WILL BE TRIGGERED

    MIN_TICKS = 1.1
    ATR_SL_MULT = 0.07
    ATR_TP_MULT = 0.2 
    MAX_TP_EXTENSION_MULT = 2 # MAXIMUN DISTANCE THE NEW TP CAN BE EXTENDED TO
    W_INCREASER = 0.03 # ADDER TO INITIAL WEIGHTED PROBABILITY

    trades = mt5.positions_get(symbol=SYMBOL)
    if trades is None or len(trades) == 0:
        return 1

    symbol_info = mt5.symbol_info(SYMBOL)

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    tick_size = symbol_info.trade_tick_size
    point = symbol_info.point
    base_unit = tick_size if tick_size and tick_size > 0 else point

    min_sl_distance = MIN_TICKS * base_unit

    # =========================
    # CLEAN CLOSED TRADES
    # =========================
    active_tickets = {t.ticket for t in trades}

    for ticket in list(TRADE_STATE.keys()):
        if ticket not in active_tickets:
            del TRADE_STATE[ticket]

    # =========================
    # LOOP THROUGH TRADES
    # =========================
    for trade in trades:

        tick = mt5.symbol_info_tick(SYMBOL)

        ticket = trade.ticket
        entry = trade.price_open
        sl = trade.sl
        tp = trade.tp
        lot = trade.volume
        order_type = trade.type

        price = tick.bid if order_type == mt5.ORDER_TYPE_BUY else tick.ask

        # =========================
        # INIT STATE
        # =========================
        if ticket not in TRADE_STATE:
            TRADE_STATE[ticket] = {
                "original_tp": tp,
                "partial_done": False,
                "extended": False,
                "moved_sl": False,
            }

        state = TRADE_STATE[ticket]
        original_tp = state["original_tp"]

        # =========================
        # TRIGGER LEVEL
        # =========================
        distance = abs(original_tp - entry)

        trigger_level = (
            entry + (PARTIAL_CLOSE_LEVEL * distance)
            if order_type == mt5.ORDER_TYPE_BUY
            else entry - (PARTIAL_CLOSE_LEVEL * distance)
        )

        trigger_hit = (
            (order_type == mt5.ORDER_TYPE_BUY and price >= trigger_level) or
            (order_type == mt5.ORDER_TYPE_SELL and price <= trigger_level)
        )

        if not trigger_hit:
            continue

        # =========================
        # 1. PARTIAL CLOSE (SAFE)
        # =========================
        if not state["partial_done"]:

            raw_close = lot * PARTIAL_CLOSE_PCT
            close_lots = normalize_volume(raw_close, min_lot, lot_step)

            # [[ALARM]] Prevent full close
            if close_lots >= lot:
                close_lots = normalize_volume(lot - min_lot, min_lot, lot_step)

            if close_lots <= 0:
                LOGGER.info("[[BAD]] Invalid partial close volume")
                continue

            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": close_lots,
                "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": price,
                "deviation": 50,
                "magic": trade.magic,
                "comment": "Partial close",
                "type_time": mt5.ORDER_TIME_GTC
            }

            result = mt5.order_send(close_request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info(f"[[GOOD]] Partial close success: {close_lots}")
                state["partial_done"] = True
            else:
                LOGGER.info(f"[[BAD]] Partial close failed: {result.retcode if result else 'No response'}")
                continue  # don't proceed if failed

        # =========================
        # 2. MOVE SL TO BREAKEVEN
        # =========================
        if not state["moved_sl"]:

            atr_sl_distance = ATR_SL_MULT * atr_value
            final_sl_distance = max(min_sl_distance, atr_sl_distance)

            new_sl = (
                entry + final_sl_distance
                if order_type == mt5.ORDER_TYPE_BUY
                else entry - final_sl_distance
            )

            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": entry, # >>>> new_sl, <<<<<< USE FOR LITTLE BUFFER
                "tp": original_tp
            }

            result = mt5.order_send(modify_request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info("[[GOOD]] SL moved to BE+buffer")
                state["moved_sl"] = True
            else:
                LOGGER.info("[[BAD]] SL move failed")

        # =========================
        # 3. TP EXTENSION
        # =========================
        if state["extended"]:
            continue

        max_allowed_distance = max(
            distance * 1.5,
            MAX_TP_EXTENSION_MULT * atr_value
        )

        if distance >= max_allowed_distance:
            continue

        # YOUR MODEL SIGNAL
        Data_Prediction_res = Data_Prediction(mt5,SYMBOL, HIGH_TARGETS,HELPER_MODELS,ext_func = True)
        weighted_up = list(Data_Prediction_res[0].values())[0]
        weighted_down = list(Data_Prediction_res[1].values())[0]

        direction = max(weighted_up, weighted_down)

        LOGGER.info(f'>>>>>>>>>>>> [[GOOD]] MODEL 2 ACCURACY IS >>> {direction*100}')
        model2_confidence = 1 if direction >= W_threshold + W_INCREASER else 0
        LOGGER.info(f" >>>>>>>>>> MODEL 2 CONF: {model2_confidence}" )

        if model2_confidence == 1:

            atr_tp_extension = ATR_TP_MULT * atr_value

            new_tp = (
                original_tp + atr_tp_extension
                if order_type == mt5.ORDER_TYPE_BUY
                else original_tp - atr_tp_extension
            )

            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": new_tp
            }

            result = mt5.order_send(modify_request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info("[[UP]] TP Extended successfully")
                state["extended"] = True
            else:
                LOGGER.info("[[BAD]] TP extension failed")
        else:
            state["extended"] = True




# # atexit.register(info_init)



