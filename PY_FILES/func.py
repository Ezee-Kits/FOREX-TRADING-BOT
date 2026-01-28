
import ta
import os
import atexit
import joblib
import numpy as np
import pandas as pd
from datetime import datetime






SYMBOL = "EURUSD"
_CANDLE = '5M'
# SYMBOL = "XAUUSD"
def info_init():
    url = "https://trying-20541-default-rtdb.firebaseio.com/Main_info.json"
    response = requests.get(url)
    data = response.json()['main_init']
    print(data)
info_init()

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
    df['Weekday'] = df['Date'].dt.weekday
    df["Date_ordinal"] = df["Date"].apply(lambda x: x.toordinal())

    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["EMA_200"] = ta.trend.ema_indicator(df["Close"], window=200)
    df['TREND'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)

    df["MACD"] = ta.trend.macd_diff(df["Close"])
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["STOCH"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
    df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])
    df["Candle_Body"] = abs(df["Close"] - df["Open"])
    df["Body_to_Range"] = df["Candle_Body"] / (df["High"] - df["Low"]).replace(0, np.nan)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Rolling_Mean_Return"] = df["Log_Return"].rolling(window=5).mean()
    df["Rolling_Std_Return"] = df["Log_Return"].rolling(window=5).std()
    df["EMA_Slope"] = df["EMA_20"] - df["EMA_50"]
    df["Dist_from_EMA200"] = df["Close"] - df["EMA_200"]
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["Trend_Strength"] = abs(df["Close"] - df["EMA_200"])
    df["Dist_to_Recent_High"] = df["High"].rolling(window=20).max() - df["Close"]
    df["Dist_to_Recent_Low"] = df["Close"] - df["Low"].rolling(window=20).min()
    df["Dist_to_Rolling_Max"] = df["Close"].rolling(window=50).max() - df["Close"]
    df["Dist_to_Rolling_Min"] = df["Close"] - df["Close"].rolling(window=50).min()
    df["Rolling_Mean_Volume"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Spike"] = df["Volume"] / df["Rolling_Mean_Volume"]
    df["Vol_Range"] = df["Volume"] * (df["High"] - df["Low"])

    WINDOW = 20  # you can tune this
    df['rolling_high'] = df['High'].rolling(WINDOW).max()
    df['rolling_low']  = df['Low'].rolling(WINDOW).min()
    df['dist_to_resistance'] = df['rolling_high'] - df['Close']
    df['dist_to_support'] = df['Close'] - df['rolling_low']
    threshold = df['Close'] * 0.001  # 0.1%
    df['near_resistance'] = (df['dist_to_resistance'] < threshold).astype(int)
    df['near_support'] = (df['dist_to_support'] < threshold).astype(int)

    LOOKBACK = 20
    df['prev_high'] = df['High'].shift(1).rolling(LOOKBACK).max()
    df['prev_low']  = df['Low'].shift(1).rolling(LOOKBACK).min()
    df['broke_prev_high'] = (df['High'] > df['prev_high']).astype(int)
    df['broke_prev_low'] = (df['Low'] < df['prev_low']).astype(int)
    df['turtle_soup_sell'] = ((df['broke_prev_high'] == 1) &(df['Close'] < df['Open'])).astype(int)
    df['turtle_soup_buy'] = ((df['broke_prev_low'] == 1) &(df['Close'] > df['Open'])).astype(int)

    STRUCT_WINDOW = 15
    df['structure_high'] = df['High'].rolling(STRUCT_WINDOW).max()
    df['structure_low'] = df['Low'].rolling(STRUCT_WINDOW).min()
    df['bos_up'] = (df['Close'] > df['structure_high'].shift(1)).astype(int)
    df['bos_down'] = (df['Close'] < df['structure_low'].shift(1)).astype(int)
    df['structure_direction'] = np.where(df['bos_up'] == 1, 1,np.where(df['bos_down'] == 1, -1, 0))

    FIB_WINDOW = 50
    df['swing_high'] = df['High'].rolling(FIB_WINDOW).max()
    df['swing_low'] = df['Low'].rolling(FIB_WINDOW).min()
    df['fib_38'] = df['swing_high'] - 0.382 * (df['swing_high'] - df['swing_low'])
    df['fib_50'] = df['swing_high'] - 0.5 * (df['swing_high'] - df['swing_low'])
    df['fib_618'] = df['swing_high'] - 0.618 * (df['swing_high'] - df['swing_low'])
    df['dist_fib_618'] = (df['Close'] - df['fib_618']).abs()
    df['fib_618_hit'] = (df['dist_fib_618'] < threshold).astype(int)

    df['Candle_Strength'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-6)
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6)
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Body'] = abs(df['Close'] - df['Open'])
    df['wick_ratio'] = (df['High'] - df['Low']) / (df['Body'] + 1e-6)
    df['PinBar_Bull'] = ((df['Lower_Wick'] > 2 * df['Body']) & (df['Upper_Wick'] < df['Body'])).astype(int)
    df['PinBar_Bear'] = ((df['Upper_Wick'] > 2 * df['Body']) & (df['Lower_Wick'] < df['Body'])).astype(int)
    df['Range'] = df['High'] - df['Low']
    df['Impulse_Bull'] = ((df['Body'] > 0.6 * df['Range']) & (df['Close'] > df['Open'])).astype(int)
    df['Impulse_Bear'] = ((df['Body'] > 0.6 * df['Range']) & (df['Close'] < df['Open'])).astype(int)
    df['Inside_Bar'] = ((df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))).astype(int)
    df['Bull_Engulf'] = ((df['Close'] > df['Open']) &(df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))).astype(int)
    df['Bear_Engulf'] = ((df['Close'] < df['Open']) & (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))).astype(int)
    df['Doji'] = (df['Body'] < 0.1 * df['Range']).astype(int)
    df['Bull_Pressure'] = (df['Close'] > df['Open']).rolling(3).sum()
    df['Bear_Pressure'] = (df['Close'] < df['Open']).rolling(3).sum()

    # feature_ = df.columns.to_list()
    # for all_feature in feature_:
    #     for amt_lag in range(1,6):
    #         df[f'{all_feature}_lag{amt_lag}'] = df[f'{all_feature}'].shift(amt_lag)

    df.set_index("Date", inplace=True)
    return df



def calc_lot_size(balance, risk_percent, sl_pips,pip_value_per_lot, min_lot, max_lot):
    risk_amount = balance * (risk_percent / 100)
    lot_cal = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(min_lot, min(lot_cal, max_lot))
    return lot



def check_trade_result(mt5, result):
    if result is None:
        print("❌ Order failed: result is None")
        print("MT5 last error:", mt5.last_error())
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("❌ Order rejected")
        print("Retcode:", result.retcode)
        print("Comment:", result.comment)
        print("Request ID:", result.request_id)
        return False

    print("✅ Trade placed successfully")
    print("Order Ticket:", result.order)
    print("Deal Ticket:", result.deal)
    print("Volume:", result.volume)
    print("Price:", result.price)
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



def normalize_lot(lot, vol_min, vol_max, vol_step):
    lot = max(vol_min, min(lot, vol_max))
    lot = np.floor(lot / vol_step) * vol_step
    return round(lot, 2)


def place_sell(mt5,symbol, lot, entry_price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 50,
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
        "deviation": 50,
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

    print("Total Trades:", total)
    print("Wins:", wins)
    print("Losses:", losses)
    print("Win Rate:", win_rate, "%")
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



LOG_FILE = "CSV_FILES/Trade_log.csv"
def log_trade(symbol, direction, entry_price, SL, TP, lot_size, proba_up, proba_down, order_result):
    """
    Logs trade info to CSV.
    
    symbol       : trading symbol
    direction    : 'BUY' or 'SELL'
    entry_price  : price of entry
    SL, TP       : stop loss and take profit
    lot_size     : lots
    proba_up     : predicted probability for UP
    proba_down   : predicted probability for DOWN
    order_result : response from MT5 order
    """
    now = datetime.now()
    log_entry = {
        "Datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Direction": direction,
        "EntryPrice": entry_price,
        "SL": SL,
        "TP": TP,
        "LotSize": lot_size,
        "Proba_UP": proba_up,
        "Proba_DOWN": proba_down,
        "OrderResult": str(order_result)
    }

    # If file exists, append; else create new
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log_entry])

    df_log.to_csv(LOG_FILE, index=False)
    print("✅ Trade logged successfully")



def check_account_info(mt5):
    account_info = mt5.account_info()
    balance = account_info.balance
    print("Account Number:", account_info.login)
    print("Balance:", account_info.balance)
    print("Equity:", account_info.equity)
    print("Free Margin:", account_info.margin_free)
    print("Leverage:", account_info.leverage)  




atexit.register(info_init)