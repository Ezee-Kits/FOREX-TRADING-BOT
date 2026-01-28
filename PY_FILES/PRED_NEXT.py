

import ta
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from func import apply_features,calc_lot_size,place_buy,place_sell






SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M30
N_BARS = 2000

if not mt5.initialize():
    raise RuntimeError("❌ MT5 initialization failed")

rates = mt5.copy_rates_from_pos(
    SYMBOL,
    TIMEFRAME,
    1,          # ← VERY IMPORTANT (skip running candle)
    N_BARS
)

if rates is None or len(rates) < N_BARS:
    mt5.shutdown()
    raise RuntimeError("❌ Failed to fetch enough closed candles")

data = pd.DataFrame(rates)
data['Date'] = pd.to_datetime(data['time'], unit='s')

data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume'
}, inplace=True)

new_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
new_df.sort_values('Date', inplace=True)
new_df.reset_index(drop=True, inplace=True)
# assert len(df) == N_BARS
# assert df.isnull().sum().sum() == 0
# mt5.shutdown()

print("✅ Last closed candle:", new_df.iloc[-1])
print(new_df.tail())

df = apply_features(new_df)
df.dropna(inplace=True)


bundle = joblib.load("ALL_MODELS/EURUSD_lgbm_bundle.pkl")
model = bundle["model"]
feature_columns = bundle["features"]
next_candle = df.loc[:, feature_columns].tail(1)
print('NEXT CANDLE DF \n',next_candle)


previos_candle = df.loc[:, feature_columns].tail(5)
previos_pred = model.predict(previos_candle)
previos_proba = model.predict_proba(previos_candle)
previos_res_df = df[['Open', 'High', 'Low', 'Close']].tail(5).copy()
previos_res_df['Prediction'] = previos_pred
previos_res_df['ACT_Close'] = previos_res_df['Close'].shift(-1)
# previos_res_df['Actual'] = previos_res_df['ACT_Close']  > previos_res_df['Close']
previos_res_df['Probability_DOWN'] = previos_proba[:,0]*100
previos_res_df['Probability_UP'] = previos_proba[:,1]*100





print('\n PREVIOUS 5 CANDLES PREDICTION \n')
print(previos_res_df)



assert set(feature_columns) == set(next_candle.columns), "Feature mismatch!" 
y_pred = model.predict(next_candle)
proba = model.predict_proba(next_candle)
print(y_pred,proba)
accu = proba[:,0]*100, proba[:,1]*100
res = f'UP MOVEMENT WITH ACCURACY : {round(accu[1][0], 2)}%' if y_pred[0]==1 else f'DOWN MOVEMENT WITH ACCURACY : {round(accu[0][0], 2)}%'
print("✅ PREDICTION FOR NEXT CANDLE:", res)


# 2️⃣ Get account info
account_info = mt5.account_info()
balance = account_info.balance
print("Account Number:", account_info.login)
print("Balance:", account_info.balance)
print("Equity:", account_info.equity)
print("Free Margin:", account_info.margin_free)
print("Leverage:", account_info.leverage)  

tick = mt5.symbol_info_tick(SYMBOL)
buy_price = tick.ask   # BUY uses ASK
sell_price = tick.bid  # SELL uses BID
risk_percent = 2  # 2% risk per trade

# 3️⃣ Calculate ATR-based SL and TP
row = df.iloc[-1]
ATR_pips = row["ATR"] / 0.0001
SL_pips = ATR_pips * 1.5
TP_pips = ATR_pips * 2.5

# BUY
SL_buy = buy_price - SL_pips * 0.0001
TP_buy = buy_price + TP_pips * 0.0001

# SELL
SL_sell = sell_price + SL_pips * 0.0001
TP_sell = sell_price - TP_pips * 0.0001
print(f"BUY Price: {buy_price}, SL: {SL_buy}, TP: {TP_buy}")
print(f"SELL Price: {sell_price}, SL: {SL_sell}, TP: {TP_sell}")

lot_size = calc_lot_size(balance,risk_percent,SL_pips,pip_value_per_lot=10,min_lot=0.01,max_lot=2)
print('LOT SIZE IS :',lot_size)


# THRESHOLD = 55  # %
# prob_down = proba[0][0] * 100
# prob_up   = proba[0][1] * 100
# if y_pred[0] == 1 and prob_up >= THRESHOLD:
#     print("Placing BUY order...")
#     result = place_buy(mt5,SYMBOL, lot_size, buy_price, SL_buy, TP_buy)
#     print("BUY order result:", result)

# if y_pred[0] == 0 and prob_down >= THRESHOLD:
#     print("Placing SELL order...")
#     result = place_sell(mt5,SYMBOL, lot_size, sell_price, SL_sell, TP_sell)
#     print("SELL order result:", result)
