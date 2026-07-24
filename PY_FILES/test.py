
# #MSE: 0.000369, MAE: 0.015496

# # # -----------------------------
# # # Plotting
# # # -----------------------------
# # add_plots = [
# #     # EMAs
# #     mpf.make_addplot(df["EMA_20"], color="orange"),
# #     mpf.make_addplot(df["EMA_50"], color="blue"),
# #     mpf.make_addplot(df["EMA_200"], color="red"),

# #     # Bollinger Bands
# #     mpf.make_addplot(df["BB_H"], color="grey"),
# #     mpf.make_addplot(df["BB_L"], color="grey"),

# #     # VWAP
# #     mpf.make_addplot(df["VWAP"], color="purple"),

# #     # RSI & Stochastic
# #     mpf.make_addplot(df["RSI"], panel=1, color="green", ylabel="RSI"),
# #     mpf.make_addplot(df["STOCH"], panel=1, color="brown"),

# #     # MACD
# #     mpf.make_addplot(df["MACD"], panel=2, color="black", ylabel="MACD"),

# #     # ATR
# #     mpf.make_addplot(df["ATR"], panel=3, color="darkred", ylabel="ATR")
# # ]

# # mpf.plot(
# #     df,
# #     type="candle",
# #     style="yahoo",
# #     title="Trading Chart (Trend, Momentum, Volatility)",
# #     volume=True,
# #     addplot=add_plots,
# #     panel_ratios=(6,2,2,2),
# #     figsize=(16,10)
# # )

# # print(48338*0.85)




# # info = mt5.symbol_info(SYMBOL)
# # print("Filling modes supported:")
# # print("FOK:", bool(info.filling_mode & mt5.ORDER_FILLING_FOK))
# # print("IOC:", bool(info.filling_mode & mt5.ORDER_FILLING_IOC))
# # print("RETURN:", bool(info.filling_mode & mt5.ORDER_FILLING_RETURN))


# # import pandas as pd
# # import numpy as np
# # from func import create_targets,SYMBOL,apply_features

# # backtest_data = pd.read_csv(f"CSV_FILES/MT5_5M_BT_{SYMBOL}_Dataset.csv")
# # backtest_df = apply_features(backtest_data)
# # backtest_df.dropna(inplace=True)
# # print(backtest_df['ATR'])

# # data = {
# #     "Close": [1.1000, 1.1005, 1.1002, 1.1008, 1.1006, 1.1010, 1.1009, 1.1012]
# # }
# # df = pd.DataFrame(data)
# # print(df)

# # df_targets = create_targets(df)
# # print(df_targets)
# # print(SYMBOL)


# # a = 7,8
# # print(a[0])


# # AVERAGE UP MOVE ACCURACY : 46.93%
# # AVERAGE DOWN MOVE ACCURACY : 53.07%
# # Account Number: 10008981518
# # Balance: 99805.89
# # Equity: 100423.89
# # Free Margin: 79979.75
# # Leverage: 100
# # Balance: 99805.89
# # Balance: 99805.89
# # Equity: 100423.89
# # Free Margin: 79979.75
# # Leverage: 100
# # pip_size: 0.0001
# # pip_value_per_lot: 10.0
# # ask_price: 1.1844000000000001
# # bid_price: 1.18438
# # spread: 2.0000000000131024e-05
# # ATR in pips: 3.6664814017751093
# # Calculated SL: 5.499722102662664 pips, TP: 16.49916630798799 pips
# # BUY Price: 1.1844000000000001, SL: 1.183850027789734, TP: 1.186049916630799
# # SELL Price: 1.18438, SL: 1.1849299722102662, TP: 1.1827300833692012
# # LOT SIZE IS: 2
# # 📌 Symbol volume rules: {'min': 0.01, 'max': 500.0, 'step': 0.01}
# # ✅ Normalized lot size: 2.0
# # PS C:\Users\HP\Desktop\PYTHON FILES\#PYTHON\FOREX TRADING>

# # def a():
# #     print('a')
# #     def b():
# #         print('b')
# #     return b  # Return the function without parentheses

# # my_function = a()  # Prints 'a' and stores 'b' in my_function
# # my_function()      # Now this prints 'b'

        
# # def a():
# #     print('a')
# #     def b():
# #         print('b')
# #     return b

# # main_func = a()
# # print(main_func.b())


# # has_open_trade = False
# # if not has_open_trade:
# #     print('yes')

# # API_KEY = "8d8759255dmsh61160898fd4a217p15aa01jsn827728277dd8" 
# # a = 9
# # b =6
# # print(max(a,b)>=4)



# #============================== BACKTESTING SCRIPT =======================#
# # main_res = []
# # for target in all_target:
# #     bundle = joblib.load(f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl")
# #     model = bundle["model"]
# #     feature_columns = bundle["features"]

# #     results = trade_backtest(df=backtest_df, model=model, feature_cols=feature_columns,threshold=55   )
# #     print(f"Backtest Results for target {target}:")
# #     analysis = analyze_results(results)
# #     main_res.append(analysis)
# #     print(analysis)
# # print(main_res)
# #============================== BACKTESTING SCRIPT =======================#


# # # for target in df.columns.to_list():
# # #     # print(f' {target} >>>>  {df[target].to_list().count(1) } out of len {len(df)}' )
# # #     print(df[target].value_counts(normalize=True))


# # a = ['icon', 'icon--ff-impact-yel']
# # found = [x for x in a if "ff-impact-yel" in x]
# # print(found)

import pandas as pd
import numpy as np
# # import ta

# # def apply_features(df):
# #     """
# #     Adds technical indicators and strategy-based features for LGBM modeling.
# #     Handles:
# #     1) Trendline breakout
# #     2) Fan principle (trend reversal angle)
# #     3) Three MA breakout
# #     4) Divergence strategy (RSI/MACD)
# #     5) Volume candlestick strategy
# #     """

# #     df = df.copy()

# #     # ==========================
# #     # Basic TA indicators
# #     # ==========================
# #     df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
# #     df['EMA10'] = ta.trend.ema_indicator(df['Close'], window=10)
# #     df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
# #     df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
# #     df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
# #     df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
# #     df['MACD'] = ta.trend.macd_diff(df['Close'])
# #     df['SMA10'] = ta.trend.sma_indicator(df['Close'], window=10)
# #     df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
# #     df['Volume_change'] = df['Volume'].pct_change()

# #     # ==========================
# #     # 1) Trendline breakout (high/low swing detection)
# #     # ==========================
# #     # Pivot points
# #     df['pivot_high'] = (df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])
# #     df['pivot_low'] = (df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])
# #     # Trendline distance (simplified)
# #     df['dist_to_prev_high'] = df['Close'] - df['High'].rolling(5).max()
# #     df['dist_to_prev_low'] = df['Close'] - df['Low'].rolling(5).min()

# #     # ==========================
# #     # 2) Fan principle (trend reversal)
# #     # ==========================
# #     # Use slope of short-term trend as proxy for fan angles
# #     df['slope_ema5'] = df['EMA5'] - df['EMA5'].shift(1)
# #     df['slope_ema10'] = df['EMA10'] - df['EMA10'].shift(1)

# #     # ==========================
# #     # 3) Three MA breakout
# #     # ==========================
# #     df['ema5_10_cross'] = (df['EMA5'] > df['EMA10']).astype(int)
# #     df['ema10_20_cross'] = (df['EMA10'] > df['EMA20']).astype(int)
# #     df['ema5_20_cross'] = (df['EMA5'] > df['EMA20']).astype(int)
# #     df['ma_alignment'] = ((df['EMA5'] > df['EMA10']) & (df['EMA10'] > df['EMA20'])).astype(int)

# #     # ==========================
# #     # 4) Divergence strategy
# #     # ==========================
# #     df['rsi_diff'] = df['RSI'] - df['RSI'].shift(1)
# #     df['price_diff'] = df['Close'] - df['Close'].shift(1)
# #     # Divergence flag: price up but RSI down → bearish, price down but RSI up → bullish
# #     df['divergence_flag'] = ((df['price_diff'] > 0) & (df['rsi_diff'] < 0)) | ((df['price_diff'] < 0) & (df['rsi_diff'] > 0))
# #     df['divergence_flag'] = df['divergence_flag'].astype(int)

# #     # ==========================
# #     # 5) Volume candlestick strategy
# #     # ==========================
# #     # Candle body size and direction
# #     df['candle_body'] = abs(df['Close'] - df['Open'])
# #     df['candle_dir'] = (df['Close'] > df['Open']).astype(int)
# #     # Volume spike: current volume > rolling mean volume
# #     df['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
# #     # Combined candle + volume signal
# #     df['vol_candle_signal'] = ((df['candle_dir'] == 1) & (df['volume_spike'] == 1)).astype(int)

# #     # Fill NaNs
# #     df.fillna(0, inplace=True)

# #     return df


# # print(max([1,2,3]))
# # print(min([1,2,3]))

# # def apply_features(df):
# #     df = df.copy()
# #     df["Date"] = pd.to_datetime(df["Date"])
# #     df = df.sort_values("Date").reset_index(drop=True)
# #     df["Close"] = pd.to_numeric(df["Close"])
# #     df["High"] = pd.to_numeric(df["High"])
# #     df["Low"] = pd.to_numeric(df["Low"])
# #     df["Open"] = pd.to_numeric(df["Open"])
# #     df["Volume"] = pd.to_numeric(df["Volume"])
# #     df['Hour'] = df['Date'].dt.hour
# #     df['Weekday'] = df['Date'].dt.weekday
# #     df["Date_ordinal"] = df["Date"].apply(lambda x: x.toordinal())

# #     df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
# #     df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
# #     bb = ta.volatility.BollingerBands(df["Close"], window=20)
# #     df["BB_H"] = bb.bollinger_hband()
# #     df["BB_L"] = bb.bollinger_lband()
# #     df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
# #     df["STOCH"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
# #     df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])
# #     df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
# #     df['EMA10'] = ta.trend.ema_indicator(df['Close'], window=10)
# #     df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
# #     df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
# #     df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)
# #     df['TREND'] = np.where(df['EMA50'] > df['EMA200'], 1, -1)
# #     df['trend_strength'] = (df['EMA50'] - df['EMA200']) / df['Close']
# #     df['SMA5'] = ta.trend.sma_indicator(df['Close'], window=5)
# #     df['SMA10'] = ta.trend.sma_indicator(df['Close'], window=10)
# #     df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
# #     df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
# #     df['SMA100'] = ta.trend.sma_indicator(df['Close'], window=100)
# #     df["WMA_10"] = ta.trend.wma_indicator(df["Close"], window=10)
# #     df["WMA_20"] = ta.trend.wma_indicator(df["Close"], window=20)
# #     df["WMA_30"] = ta.trend.wma_indicator(df["Close"], window=30)
# #     df["WMA_50"] = ta.trend.wma_indicator(df["Close"], window=50)
# #     df["MACD"] = ta.trend.macd(df["Close"])
# #     df["MACD_signal"] = ta.trend.macd_signal(df["Close"])
# #     df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
# #     df["+DI"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"])
# #     df["-DI"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"])
# #     df["Ichimoku_A"] = ta.trend.ichimoku_a(df["High"], df["Low"])
# #     df["Ichimoku_B"] = ta.trend.ichimoku_b(df["High"], df["Low"])
# #     df["TRIX"] = ta.trend.trix(df["Close"])
# #     df["KST"] = ta.trend.kst(df["Close"])
# #     df["Stoch_RSI"] = ta.momentum.stochrsi(df["Close"])
# #     # df["CCI"] = ta.momentum.cci(df["High"], df["Low"], df["Close"])
# #     df["ROC"] = ta.momentum.roc(df["Close"])
# #     df["AO"] = ta.momentum.awesome_oscillator(df["High"], df["Low"])
# #     df["Donchian_high"] = ta.volatility.donchian_channel_hband(df["High"], df["Low"], df["Close"])
# #     df["Donchian_low"] = ta.volatility.donchian_channel_lband(df["High"], df["Low"], df["Close"])
# #     df["Keltner_high"] = ta.volatility.keltner_channel_hband(df["High"], df["Low"], df["Close"])
# #     df["Keltner_low"] = ta.volatility.keltner_channel_lband(df["High"], df["Low"], df["Close"])
# #     df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
# #     df["MFI"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"])
# #     ema = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
# #     ema_of_ema = ta.trend.EMAIndicator(close=ema, window=20).ema_indicator()
# #     df['DEMA_20'] = 2 * ema - ema_of_ema
# #     ema2 = ema_of_ema
# #     ema3 = ta.trend.EMAIndicator(close=ema2, window=20).ema_indicator()
# #     df['TEMA_20'] = 3 * ema - 3 * ema2 + ema3
# #     df['KAMA_20'] = ta.momentum.KAMAIndicator(close=df['Close'], window=20).kama()
# #     df['Volume_change'] = df['Volume'].pct_change().rolling(3).mean()
# #     df["AD"] = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])

# #     df['ATR_MEAN'] = df['ATR'].rolling(50).mean()
# #     df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
# #     df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
# #     df['pivot_high'] = ((df['High'] > df['High'].shift(1)) &(df['High'] > df['High'].shift(2))).shift(2).fillna(0).astype(int)
# #     df['pivot_low'] = (((df['Low'] < df['Low'].shift(1)) &df['Low'] < df['Low'].shift(2))).shift(2).fillna(0).astype(int)
# #     df['dist_to_prev_high'] = (df['Close'] - df['High'].rolling(5).max()) / df['ATR']
# #     df['dist_to_prev_low'] = (df['Close'] - df['Low'].rolling(5).min()) / df['ATR']
# #     df['slope_ema5']  = (df['EMA5'] - df['EMA5'].shift(1)) / df['Close']
# #     df['slope_ema10'] = (df['EMA10'] - df['EMA10'].shift(1)) / df['Close']
# #     df['ema5_10_state'] = (df['EMA5'] > df['EMA10']).astype(int)
# #     df['ema10_20_state'] = (df['EMA10'] > df['EMA20']).astype(int)
# #     df['ema5_20_state'] = (df['EMA5'] > df['EMA20']).astype(int)
# #     df['ema5_10_cross_event'] = ((df['EMA5'] > df['EMA10']) &(df['EMA5'].shift(1) <= df['EMA10'].shift(1))).astype(int)
# #     df['ma_alignment'] = ((df['EMA5'] > df['EMA10']) &(df['EMA10'] > df['EMA20'])).astype(int)
# #     df['price_diff'] = (df['Close'] - df['Close'].shift(1)) / df['Close']
# #     df['rsi_diff'] = df['RSI'] - df['RSI'].shift(1)
# #     df['divergence_flag'] = (
# #         (df['price_diff'].rolling(3).mean() > 0) &
# #         (df['rsi_diff'].rolling(3).mean() < 0)) | ((df['price_diff'].rolling(3).mean() < 0) &(df['rsi_diff'].rolling(3).mean() > 0))
# #     df['divergence_flag'] = df['divergence_flag'].astype(int)
# #     df['candle_body'] = abs(df['Close'] - df['Open']) / df['Close']
# #     df['candle_dir'] = (df['Close'] > df['Open']).astype(int)
# #     vol_mean = df['Volume'].rolling(20).mean()
# #     vol_std  = df['Volume'].rolling(20).std()
# #     df['volume_spike'] = (df['Volume'] > (vol_mean + vol_std)).astype(int)
# #     df['vol_candle_signal'] = ((df['candle_dir'] == 1) &(df['volume_spike'] == 1)).astype(int)

# #     df['Body_to_Range'] = (df['candle_body'] /(df['High'] - df['Low']).replace(0, np.nan))
# #     df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
# #     df['Rolling_Mean_Return'] = df['Log_Return'].rolling(5).mean()
# #     df['Rolling_Std_Return'] = df['Log_Return'].rolling(5).std()
# #     df['EMA_20_50_dist'] = (df['EMA20'] - df['EMA50']) / df['ATR']
# #     df['Dist_from_EMA200'] = (df['Close'] - df['EMA200']) / df['ATR']
# #     df['Trend_Strength'] = abs(df['Close'] - df['EMA200']) / df['ATR']
 
# #     df['Dist_to_Recent_High'] = (df['High'].rolling(20).max() - df['Close']) / df['ATR']
# #     df['Dist_to_Recent_Low'] = (df['Close'] - df['Low'].rolling(20).min()) / df['ATR']
# #     df['Dist_to_Rolling_Max'] = (df['Close'].rolling(50).max() - df['Close']) / df['ATR']
# #     df['Dist_to_Rolling_Min'] = (df['Close'] - df['Close'].rolling(50).min()) / df['ATR']
# #     df["Rolling_Mean_Volume"] = df["Volume"].rolling(window=20).mean()
# #     df["Volume_Spike"] = df["Volume"] / df["Rolling_Mean_Volume"]
# #     df["Vol_Range"] = df["Volume"] * (df["High"] - df["Low"])
# #     df['Volume_Z'] = ((df['Volume'] - df['Rolling_Mean_Volume']) /df['Volume'].rolling(20).std()).clip(-5, 5)

# #     WINDOW = 20  # you can tune this
# #     df['rolling_high'] = df['High'].rolling(WINDOW).max().shift(1)
# #     df['rolling_low']  = df['Low'].rolling(WINDOW).min().shift(1)
# #     df['dist_to_resistance'] = (df['rolling_high'] - df['Close']) / df['ATR']
# #     df['dist_to_support'] = (df['Close'] - df['rolling_low']) / df['ATR']
# #     df['near_resistance'] = (df['dist_to_resistance'] < 0.5).astype(int)
# #     df['near_support'] = (df['dist_to_support'] < 0.5).astype(int)

# #     LOOKBACK = 20
# #     df['prev_high'] = df['High'].shift(1).rolling(LOOKBACK).max()
# #     df['prev_low']  = df['Low'].shift(1).rolling(LOOKBACK).min()
# #     df['broke_prev_high'] = (df['High'] > df['prev_high']).astype(int)
# #     df['broke_prev_low'] = (df['Low'] < df['prev_low']).astype(int)
# #     df['turtle_soup_sell'] = ((df['High'] > df['prev_high']) &(df['Close'] < df['prev_high'])).astype(int)
# #     df['turtle_soup_buy'] = ((df['Low'] < df['prev_low']) &(df['Close'] > df['prev_low'])).astype(int)

# #     STRUCT_WINDOW = 15
# #     df['structure_high'] = df['High'].rolling(STRUCT_WINDOW).max().shift(1)
# #     df['structure_low'] = df['Low'].rolling(STRUCT_WINDOW).min().shift(1)
# #     df['bos_up'] = (df['Close'] > df['structure_high']).astype(int)
# #     df['bos_down'] = (df['Close'] < df['structure_low']).astype(int)
# #     df['structure_direction'] = np.where(df['bos_up'] == 1, 1,np.where(df['bos_down'] == 1, -1, 0))

# #     FIB_WINDOW = 50
# #     df['swing_high'] = df['High'].rolling(FIB_WINDOW).max().shift(1)
# #     df['swing_low'] = df['Low'].rolling(FIB_WINDOW).min().shift(1)
# #     fib_range = df['swing_high'] - df['swing_low']
# #     df['fib_38'] = df['swing_high'] - 0.382 * fib_range
# #     df['fib_50'] = df['swing_high'] - 0.5 * fib_range
# #     df['fib_618'] = df['swing_high'] - 0.618 * fib_range
# #     df['dist_fib_618'] = abs(df['Close'] - df['fib_618']) / df['ATR']
# #     df['fib_618_zone'] = (df['dist_fib_618'] < 0.5).astype(int)
# #     df['Range'] = df['High'] - df['Low'] + 1e-6
# #     df['Body'] = abs(df['Close'] - df['Open'])
# #     df['Candle_Strength'] = (df['Close'] - df['Open']) / df['Range']
# #     df['Close_Position'] = (df['Close'] - df['Low']) / df['Range']
# #     df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']
# #     df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']
# #     df['wick_ratio'] = df['Range'] / (df['Body'] + 1e-6)
# #     df['PinBar_Bull'] = ((df['Lower_Wick'] > 0.6) &(df['Upper_Wick'] < 0.2)).astype(int)
# #     df['PinBar_Bear'] = ((df['Upper_Wick'] > 0.6) &(df['Lower_Wick'] < 0.2)).astype(int)
# #     df['Impulse_Bull'] = ((df['Body'] / df['Range'] > 0.6) &(df['Close'] > df['Open'])).astype(int)
# #     df['Impulse_Bear'] = ((df['Body'] / df['Range'] > 0.6) &(df['Close'] < df['Open'])).astype(int)
# #     df['Inside_Bar'] = ((df['High'] < df['High'].shift(1)) &(df['Low'] > df['Low'].shift(1))).astype(int)
# #     df['Bull_Engulf'] = ((df['Close'] > df['Open']) &(df['Open'] < df['Close'].shift(1)) &(df['Close'] > df['Open'].shift(1))).astype(int)
# #     df['Bear_Engulf'] = ((df['Close'] < df['Open']) &(df['Open'] > df['Close'].shift(1)) &(df['Close'] < df['Open'].shift(1))).astype(int)
# #     df['Doji'] = (df['Body'] / df['Range'] < 0.1).astype(int)
# #     df['Bull_Pressure'] = ((df['Close'] > df['Open']).rolling(3).mean())
# #     df['Bear_Pressure'] = ((df['Close'] < df['Open']).rolling(3).mean())


# #     # feature_ = df.columns.to_list()
# #     # for all_feature in feature_:
# #     #     for amt_lag in range(1,8):
# #     #         df[f'{all_feature}_lag{amt_lag}'] = df[f'{all_feature}'].shift(amt_lag)

# #     df.set_index("Date", inplace=True)
# #     return df


# #=========================== LGBMCLASSIFIER [BINARY] ============================ 
# # data = pd.read_csv(f'CSV_FILES/MT5_5M_{SYMBOL}_Exchange_Rate_Dataset.csv') 
# # df = apply_features(data)
# # df = create_targets(df)
# # df.dropna(inplace=True)

# # all_target = ['T_5M','T_10M','T_15M','T_20M','T_30M']
# # train_df = df.copy()
# # X_full = train_df.drop(columns=all_target)

# # for target in all_target:
# #     print(f"\n🔎 Tuning model for {target}...")
# #     y = train_df[target]
# #     base_model = LGBMClassifier(n_estimators=200, random_state=42)
# #     base_model.fit(X_full, y)
# #     importance = base_model.feature_importances_
# #     feature_names = X_full.columns.to_list()
# #     sort_indx = np.argsort(importance)[::-1]

# #     top_90_indx = sort_indx[:90]
# #     top90_features = [feature_names[i] for i in top_90_indx]
# #     X = X_full[top90_features]
# #     print("Top 90 features selected.")

# #     def objective(trial):
# #         params = {
# #             "n_estimators": trial.suggest_int("n_estimators", 100, 400),
# #             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
# #             "num_leaves": trial.suggest_int("num_leaves", 20, 200),
# #             "max_depth": trial.suggest_int("max_depth", 3, 12),
# #             "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
# #             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
# #             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
# #             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
# #             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
# #             "random_state": 42,
# #             "n_jobs": -1}

# #         X_train_split, X_valid, y_train_split, y_valid = train_test_split(
# #             X, y, test_size=0.2, shuffle=False  # IMPORTANT for time series
# #         )

# #         model = LGBMClassifier(**params)
# #         model.fit(
# #             X_train_split,
# #             y_train_split,
# #             eval_set=[(X_valid, y_valid)],
# #             eval_metric="auc"
# #         )
# #         preds = model.predict_proba(X_valid)[:, 1]
# #         score = roc_auc_score(y_valid, preds)
# #         return score

# #     study = optuna.create_study(direction="maximize")
# #     study.optimize(objective, n_trials=15)

# #     print("Best Score:", study.best_value)
# #     print("Best Params:", study.best_params)

# #     best_model = LGBMClassifier(
# #         **study.best_params,
# #         random_state=42,
# #         n_jobs=-1
# #     )

# #     best_model.fit(X, y)

# #     joblib.dump(
# #         {
# #             "model": best_model,
# #             "features": top90_features,
# #             "best_params": study.best_params,
# #             "best_score": study.best_value
# #         },
# #         f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl"
# #     )

# #     print(f"✅ Optimized model for {target} saved.")
# # print("------------------------------------------------")









# # #####=========================== LGBMREGRESSOR [REGRESSOR] ============================ 
# # data = pd.read_csv(f'CSV_FILES/MT5_5M_{SYMBOL}_Exchange_Rate_Dataset.csv') 
# # df = apply_features(data)
# # df = create_reg_targets(df)
# # df.dropna(inplace=True)

# # all_target = ['TRE_5M','TRE_10M','TRE_15M','TRE_20M','TRE_30M']
# # train_df = df.copy()
# # X_full = train_df.drop(columns=all_target)

# # for target in all_target:
# #     print(f"\n🔎 Tuning model for {target}...")
# #     y = train_df[target]
# #     base_model = LGBMClassifier(n_estimators=200, random_state=42)
# #     base_model.fit(X_full, y)
# #     importance = base_model.feature_importances_
# #     feature_names = X_full.columns.to_list()
# #     sort_indx = np.argsort(importance)[::-1]

# #     top_90_indx = sort_indx[:90]
# #     top90_features = [feature_names[i] for i in top_90_indx]
# #     X = X_full[top90_features]
# #     print("Top 90 features selected.")

# #     def objective(trial):
# #         params = {
# #             "n_estimators": trial.suggest_int("n_estimators", 100, 400),
# #             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
# #             "num_leaves": trial.suggest_int("num_leaves", 20, 200),
# #             "max_depth": trial.suggest_int("max_depth", 3, 12),
# #             "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
# #             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
# #             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
# #             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
# #             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
# #             "random_state": 42,
# #             "n_jobs": -1}

# #         X_train_split, X_valid, y_train_split, y_valid = train_test_split(
# #             X, y, test_size=0.2, shuffle=False  # IMPORTANT for time series
# #         )

# #         model = LGBMClassifier(**params)
# #         model.fit(
# #             X_train_split,
# #             y_train_split,
# #             eval_set=[(X_valid, y_valid)],
# #             eval_metric="auc"
# #         )
# #         preds = model.predict_proba(X_valid)[:, 1]
# #         score = roc_auc_score(y_valid, preds)
# #         return score

# #     study = optuna.create_study(direction="maximize")
# #     study.optimize(objective, n_trials=15)

# #     print("Best Score:", study.best_value)
# #     print("Best Params:", study.best_params)

# #     best_model = LGBMClassifier(
# #         **study.best_params,
# #         random_state=42,
# #         n_jobs=-1
# #     )

# #     best_model.fit(X, y)

# #     joblib.dump(
# #         {
# #             "model": best_model,
# #             "features": top90_features,
# #             "best_params": study.best_params,
# #             "best_score": study.best_value
# #         },
# #         f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl"
# #     )

# #     print(f"✅ Optimized model for {target} saved.")
# # print("------------------------------------------------")



# ### OLD TRAINING HORIZON CLASSIFICATION
# # tp_mult = 1.2
# # sl_mult = 0.8

# # res5M_list = []
# # res10M_list = []
# # res15M_list = []
# # res20M_list = []
# # res30M_list = []
# # res35M_list = []
# # res45M_list = []
# # res55M_list = []
# # res60M_list = []
# # for datas in range(0,len(df)-12):
# #     tp_distance = tp_mult * df['ATR'][datas] #0.00050 #
# #     sl_distance = sl_mult * df['ATR'][datas] #0.00030 #

# #     high_base = df['High'][datas]
# #     result5m = int((df['High'][datas+1] > high_base + tp_distance) and (df['Low'][datas+1]  > high_base - sl_distance))
# #     result10m = int((df['High'][datas+2] > high_base + tp_distance) and (df['Low'][datas+2]  > high_base - sl_distance))
# #     result15m = int((df['High'][datas+3] > high_base + tp_distance) and (df['Low'][datas+3]  > high_base - sl_distance))
# #     result20m = int((df['High'][datas+4] > high_base + tp_distance) and (df['Low'][datas+4]  > high_base - sl_distance))
# #     result30m = int((df['High'][datas+6] > high_base + tp_distance) and (df['Low'][datas+6]  > high_base - sl_distance))
# #     result35m = int((df['High'][datas+7] > high_base + tp_distance) and (df['Low'][datas+7]  > high_base - sl_distance))
# #     result45m = int((df['High'][datas+9] > high_base + tp_distance) and (df['Low'][datas+9]  > high_base - sl_distance))
# #     result55m = int((df['High'][datas+11] > high_base + tp_distance) and (df['Low'][datas+11]  > high_base - sl_distance))
# #     result60m = int((df['High'][datas+12] > high_base + tp_distance) and (df['Low'][datas+12]  > high_base - sl_distance))

# #     res5M_list.append(result5m)
# #     res10M_list.append(result10m)
# #     res15M_list.append(result15m)
# #     res20M_list.append(result20m)
# #     res30M_list.append(result30m)
# #     res35M_list.append(result35m)
# #     res45M_list.append(result45m)
# #     res55M_list.append(result55m)
# #     res60M_list.append(result60m)

# # df = df.iloc[:-12]
# # df['THL_5M'] = res5M_list
# # df['THL_10M'] = res10M_list
# # df['THL_15M'] = res15M_list
# # df['THL_20M'] = res20M_list
# # df['THL_30M'] = res30M_list
# # df['THL_35M'] = res35M_list
# # df['THL_45M'] = res45M_list
# # df['THL_55M'] = res55M_list
# # df['THL_60M'] = res60M_list
# # print(df)

# # print(df['THL_5M'].value_counts(normalize=True))
# # print(df['THL_10M'].value_counts(normalize=True))
# # print(df['THL_15M'].value_counts(normalize=True))
# # print(df['THL_20M'].value_counts(normalize=True))
# # print(df['THL_30M'].value_counts(normalize=True))


# # weights = {
# #     'THL_5M': 0.4,
# #     'THL_10M': 0.7,
# #     'THL_15M': 1.5,
# #     'THL_20M': 2.2,  
# #     'THL_30M': 2.8,
# #     'THL_35M': 2.6,
# #     'THL_45M': 1.8,
# #     'THL_55M': 1.4,
# #     'THL_60M': 1.2
# # }


# import os
# import ta
# import time
# import joblib
# import numpy as np
# import pandas as pd
# import MetaTrader5 as mt5
# from collections import deque
# from func import apply_features, calc_lot_size, place_buy, place_sell, move_sl_and_partial_close, SYMBOL, normalize_lot, get_symbol_volume_info, get_pip_info, log_trade

# # --------------------------- CONFIG ---------------------------
# HIGH_TARGETS = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']
# delay_time = 60*2  # 2 minutes

# # Base weights for each target
# BASE_WEIGHTS = {
#     'THL_5M' : 1.2,
#     'THL_10M' : 1.4,
#     'THL_15M' : 1.3,
#     'THL_20M' : 1.1,
#     'THL_30M' : 0.9
# }

# # Keep last N predictions for adaptive accuracy
# RECENT_N = 20
# recent_accuracy = {target: deque(maxlen=RECENT_N) for target in HIGH_TARGETS}

# # --------------------------- LOAD MODELS ---------------------------
# def load_all_models(symbol):
#     models_dict = {}
#     for target in HIGH_TARGETS:
#         file_path = f"ALL_MODELS/{symbol}_classifier_{target}.pkl"
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Model not found: {file_path}")
#         bundle = joblib.load(file_path)
#         models_dict[target] = {
#             "model": bundle["model"],
#             "features": bundle["features"]
#         }
#         print(f"✅ Loaded model for {target}")
#     return models_dict

# HIGH_MODELS = load_all_models(SYMBOL)

# # --------------------------- TRADING LOOP ---------------------------
# while True:
#     TIMEFRAME = mt5.TIMEFRAME_M5
#     N_BARS = 700

#     if not mt5.initialize():
#         raise RuntimeError("❌ MT5 initialization failed")

#     rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, N_BARS)
#     if rates is None or len(rates) < N_BARS:
#         mt5.shutdown()
#         raise RuntimeError("❌ Failed to fetch enough closed candles")

#     df = pd.DataFrame(rates)
#     df['Date'] = pd.to_datetime(df['time'], unit='s')
#     df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'}, inplace=True)
#     df = df[['Date','Open','High','Low','Close','Volume']].sort_values('Date').reset_index(drop=True)

#     df = apply_features(df)
#     df.dropna(inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     up_moves, down_moves = {}, {}
#     prev_res_df = df[['Open','Close']].tail(5).copy()

#     # ------------------- PREDICTIONS -------------------
#     for target in HIGH_TARGETS:
#         print(f'\n[[ CURRENTLY PREDICTING TARGET : {target} ]]')
#         model = HIGH_MODELS[target]["model"]
#         features = HIGH_MODELS[target]["features"]

#         # Previous candle predictions
#         prev_candles = df[features].tail(5)
#         prev_pred = model.predict(prev_candles)
#         prev_proba = model.predict_proba(prev_candles)
#         prev_res_df[target] = prev_pred
#         prev_res_df[f'{target}_DN'] = (prev_proba[:,0]*100).round(2)
#         prev_res_df[f'{target}_UP'] = (prev_proba[:,1]*100).round(2)

#         # Next candle prediction
#         next_candle = df[features].tail(1)
#         assert set(features) == set(next_candle.columns), "Feature mismatch!"
#         y_pred = model.predict(next_candle)
#         proba = model.predict_proba(next_candle)
#         up_moves[target] = round(proba[:,1][0]*100,2)
#         down_moves[target] = round(proba[:,0][0]*100,2)
#         res = f'UP {up_moves[target]}%' if y_pred[0]==1 else f'DOWN {down_moves[target]}%'
#         print(f"✅ {target} - NEXT CANDLE PREDICTION: {res}")

#         # Update recent accuracy (last candle)
#         actual = 1 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else 0
#         recent_accuracy[target].append(int(prev_pred[-1] == actual))

#     print('\nPREVIOUS 5 CANDLES PREDICTION:\n', prev_res_df)

#     # ------------------- ADAPTIVE WEIGHTS -------------------
#     adaptive_weights = {}
#     for target in HIGH_TARGETS:
#         acc = sum(recent_accuracy[target])/len(recent_accuracy[target]) if recent_accuracy[target] else 0.5
#         adaptive_weights[target] = BASE_WEIGHTS[target] * (1 + acc)

#     total_weight = sum(adaptive_weights.values())
#     adaptive_weights = {k:v/total_weight for k,v in adaptive_weights.items()}

#     weighted_up = sum(up_moves[k]/100 * adaptive_weights[k] for k in HIGH_TARGETS)
#     weighted_down = sum(down_moves[k]/100 * adaptive_weights[k] for k in HIGH_TARGETS)

#     std_up = np.std(list(up_moves.values()))/100
#     std_down = np.std(list(down_moves.values()))/100
#     agreement_up = 1 - std_up
#     agreement_down = 1 - std_down

#     # ------------------- ADAPTIVE THRESHOLDS -------------------
#     base_threshold = 0.7
#     threshold_up = base_threshold - 0.1*agreement_up - 0.05*(weighted_up)
#     threshold_up = min(max(threshold_up,0.55),0.75)
#     threshold_down = base_threshold - 0.1*agreement_down - 0.05*(weighted_down)
#     threshold_down = min(max(threshold_down,0.55),0.75)

#     print('Adaptive Weights:', adaptive_weights)
#     print('Weighted UP:', round(weighted_up,2), 'Weighted DOWN:', round(weighted_down,2))
#     print('Threshold UP:', round(threshold_up,2), 'Threshold DOWN:', round(threshold_down,2))

#     # ------------------- ACCOUNT & MARKET INFO -------------------
#     account_info = mt5.account_info()
#     balance = account_info.balance
#     tick = mt5.symbol_info_tick(SYMBOL)
#     ask_price, bid_price = tick.ask, tick.bid
#     pip_info = get_pip_info(mt5,SYMBOL)
#     pip_size, pip_value_per_lot = pip_info["pip_size"], pip_info["pip_value_per_lot"]
#     spread = ask_price - bid_price
#     symbol_info = mt5.symbol_info(SYMBOL)
#     spread_points = spread / symbol_info.point
#     if spread_points > 30:  # max spread points threshold
#         print(f"⚠️ Spread too high: {spread_points:.2f} points, skipping trade")
#         time.sleep(delay_time)
#         continue

#     # ------------------- CALCULATE SL & TP -------------------
#     row = df.iloc[-1]
#     ATR_pips = row["ATR"] / pip_size
#     SL_pips = ATR_pips * 0.8
#     TP_pips = ATR_pips * 1.2

#     entry_buy, SL_buy, TP_buy = ask_price, ask_price - SL_pips*pip_size, ask_price + TP_pips*pip_size
#     entry_sell, SL_sell, TP_sell = bid_price, bid_price + SL_pips*pip_size, bid_price - TP_pips*pip_size

#     # ------------------- LOT SIZE -------------------
#     risk_percent = 1
#     lot_size = calc_lot_size(balance=balance, risk_percent=risk_percent, sl_pips=SL_pips, pip_value_per_lot=pip_value_per_lot, min_lot=0.01, max_lot=2)
#     vol_info = get_symbol_volume_info(mt5,SYMBOL)
#     lot_size = normalize_lot(lot_size,vol_info["min"],vol_info["max"],vol_info["step"])

#     # ------------------- MOVE SL & CHECK OPEN POSITIONS -------------------
#     move_sl_and_partial_close(mt5, SYMBOL)
#     positions = mt5.positions_get(symbol=SYMBOL)
#     has_open_trade = False
#     open_trade_type = None

#     if positions is not None and len(positions) > 0:
#         has_open_trade = True
#         open_trade_type = positions[0].type

#     # ------------------- TRADE DECISION -------------------
#     if not has_open_trade:
#         if weighted_up >= threshold_up and weighted_up > weighted_down and weighted_up >= 0.9:
#             print("🚀 FINAL SIGNAL: BUY")
#             result = place_buy(mt5, SYMBOL, lot_size, entry_buy, SL_buy, TP_buy)
#             log_trade(mt5, SYMBOL, "BUY", entry_buy, SL_buy, TP_buy, lot_size, weighted_up, weighted_down, result)
#         elif weighted_down >= threshold_down and weighted_down > weighted_up and weighted_down >= 0.9:
#             print("🔻 FINAL SIGNAL: SELL")
#             result = place_sell(mt5, SYMBOL, lot_size, entry_sell, SL_sell, TP_sell)
#             log_trade(mt5, SYMBOL, "SELL", entry_sell, SL_sell, TP_sell, lot_size, weighted_up, weighted_down, result)
#         else:
#             print("⏳ FINAL SIGNAL: NO TRADE")
#     else:
#         while True:
#             status = move_sl_and_partial_close(mt5, SYMBOL)
#             if status == 1:
#                 break
#             time.sleep(1)

#         if open_trade_type == mt5.ORDER_TYPE_BUY and weighted_down >= threshold_down:
#             print("⚠️ SELL blocked: BUY trade already open")
#         elif open_trade_type == mt5.ORDER_TYPE_SELL and weighted_up >= threshold_up:
#             print("⚠️ BUY blocked: SELL trade already open")
#         else:
#             print("No new trade placed")

#     print(f"\n>>>> WAITING FOR NEXT CANDLE [{delay_time/60} MINUTES] <<<<\n")
#     time.sleep(delay_time)



# def move_sl_and_partial_close(mt5, SYMBOL):

#     PARTIAL_CLOSE_PCT = 0.4 # THAT IS 40% OF ORIGINAL LOT
#     PARTIAL_CLOSE_LEVEL = 0.4 # CLOSES 40% LOTS SIZE AT 40% FAVOUR MOVES TO TP

#     trades = mt5.positions_get(symbol=SYMBOL)

#     if trades is None or len(trades) == 0:
#         return 1

#     tick = mt5.symbol_info_tick(SYMBOL)

#     for trade in trades:

#         entry = trade.price_open
#         sl = trade.sl
#         tp = trade.tp
#         lot = trade.volume
#         order_type = trade.type
#         print('\nTP :',tp)
#         print('SL',sl)
#         print('LOT :',lot)

#         # If SL already equals entry → management already done
#         if sl == entry:
#             return 1

#         distance = abs(tp - entry)
#         print('DISTANCE BETWEEN ENTRY AND TP :',distance)

#         trigger_level = entry + (PARTIAL_CLOSE_LEVEL * distance) if order_type == 0 else entry - (PARTIAL_CLOSE_LEVEL * distance)
#         print('TRIGGER LEVEL IS :',trigger_level,'\n')
#         price = tick.bid if order_type == 0 else tick.ask

#         trigger_hit = (
#             order_type == 0 and price >= trigger_level) or (order_type == 1 and price <= trigger_level)

#         if not trigger_hit:
#             return 0

#         close_lots = round(lot * PARTIAL_CLOSE_PCT, 2)

#         close_request = {
#             "action": mt5.TRADE_ACTION_DEAL,
#             "symbol": SYMBOL,
#             "volume": close_lots,
#             "type": mt5.ORDER_TYPE_SELL if order_type == 0 else mt5.ORDER_TYPE_BUY,
#             "position": trade.ticket,
#             "price": tick.bid if order_type == 0 else tick.ask,
#             "deviation": 50,
#             "magic": trade.magic,
#             "comment": "Partial close 40%",
#             "type_time": mt5.ORDER_TIME_GTC
#         }

#         mt5.order_send(close_request)

#         modify_request = {
#             "action": mt5.TRADE_ACTION_SLTP,
#             "position": trade.ticket,
#             "sl": entry,
#             "tp": tp
#         }

#         mt5.order_send(modify_request)

#         return 1
# WITH NUMBER OF TRADES 322
# Total Wins: 117
# Total Losses: 205
# Total Gross Profit: 1718.485621234586
# Total Gross Loss: 2003.3807368465539
# Profit Factor: 0.8577928247126851

# WITH NUMBER OF TRADES 656
# Total Wins: 219
# Total Losses: 437

# # ==========================================================
# # WEIGHTS (HORIZON IMPORTANCE)
# # ==========================================================
# weights = {
#     'THL_5M': 1.3,
#     'THL_10M': 1.4,
#     'THL_15M': 1.1,
#     'THL_20M': 1.0,
#     'THL_30M': 0.9
# }

# # ==========================================================
# # MODEL TYPE WEIGHTS (VERY IMPORTANT)
# # ==========================================================
# model_weights = {
#     "main": 0.5,       # most reliable
#     "direction": 0.3,  # smoother / general trend
#     "helper": 0.2      # strict confirmation
# }

# # ==========================================================
# # STORAGE
# # ==========================================================
# up_moves = {}
# down_moves = {}

# # ==========================================================
# # PREDICTION LOOP
# # ==========================================================
# for target in HIGH_TARGETS:

#     print(f'\n[[ PREDICTING TARGET: {target} ]]')

#     # -----------------------------
#     # LOAD MODELS
#     # -----------------------------
#     main_model = MAIN_MODELS[target]["model"]
#     dir_model  = DIRECTION_MODELS[target]["model"]
#     help_model = HELPER_MODELS[target]["model"]

#     main_cols = MAIN_MODELS[target]["features"]
#     dir_cols  = DIRECTION_MODELS[target]["features"]
#     help_cols = HELPER_MODELS[target]["features"]

#     # -----------------------------
#     # PREPARE INPUTS
#     # -----------------------------
#     X_main = df.iloc[[datas]][main_cols]
#     X_dir  = df.iloc[[datas]][dir_cols]
#     X_help = df.iloc[[datas]][help_cols]

#     # Safety check
#     assert set(main_cols) == set(X_main.columns)
#     assert set(dir_cols)  == set(X_dir.columns)
#     assert set(help_cols) == set(X_help.columns)

#     # -----------------------------
#     # PREDICTIONS (PROBABILITIES)
#     # -----------------------------
#     main_proba = main_model.predict_proba(X_main)[0]
#     dir_proba  = dir_model.predict_proba(X_dir)[0]
#     help_proba = help_model.predict_proba(X_help)[0]

#     # -----------------------------
#     # COMBINE MODELS (WEIGHTED)
#     # -----------------------------
#     up_prob = (
#         main_proba[1] * model_weights["main"] +
#         dir_proba[1]  * model_weights["direction"] +
#         help_proba[1] * model_weights["helper"]
#     )

#     down_prob = (
#         main_proba[0] * model_weights["main"] +
#         dir_proba[0]  * model_weights["direction"] +
#         help_proba[0] * model_weights["helper"]
#     )

#     # Normalize (optional but safer)
#     total = up_prob + down_prob
#     up_prob /= total
#     down_prob /= total

#     up_moves[target] = round(up_prob * 100, 2)
#     down_moves[target] = round(down_prob * 100, 2)

#     direction = "UP" if up_prob > down_prob else "DOWN"

#     print(f"MAIN  : {main_proba}")
#     print(f"DIR   : {dir_proba}")
#     print(f"HELP  : {help_proba}")
#     print(f"➡ FINAL → {direction} ({round(max(up_prob, down_prob)*100,2)}%)")

# # ==========================================================
# # FINAL AGGREGATION (ACROSS HORIZONS)
# # ==========================================================
# print('\n================ FINAL SCORES =================')

# up_moves_mean = round(sum(up_moves.values()) / len(up_moves), 2)
# down_moves_mean = round(sum(down_moves.values()) / len(down_moves), 2)

# print(f'AVG UP   : {up_moves_mean}%')
# print(f'AVG DOWN : {down_moves_mean}%')

# # -----------------------------
# # WEIGHTED FINAL DECISION
# # -----------------------------
# probs_up = {k: v / 100 for k, v in up_moves.items()}
# probs_down = {k: v / 100 for k, v in down_moves.items()}

# total_weight = sum(weights.values())

# weighted_up = sum(probs_up[k] * weights[k] for k in weights) / total_weight
# weighted_down = sum(probs_down[k] * weights[k] for k in weights) / total_weight

# print('\n================ WEIGHTED RESULT =================')
# print(f'WEIGHTED UP   : {round(weighted_up*100,2)}%')
# print(f'WEIGHTED DOWN : {round(weighted_down*100,2)}%')

# final_direction = "UP" if weighted_up > weighted_down else "DOWN"
# confidence = max(weighted_up, weighted_down) * 100

# print(f'\n🚀 FINAL SIGNAL: {final_direction} ({round(confidence,2)}%)')


# ==============================
# DYNAMIC WEIGHTING SYSTEM
# ==============================

# performance_tracker = {
#     "main": [],
#     "direction": [],
#     "helper": []
# }

# WINDOW_SIZE = 50
# DECAY = 0.9


# def update_performance(performance_tracker,
#                        main_proba, dir_proba, help_proba,
#                        actual):

#     main_score = main_proba[actual]
#     dir_score  = dir_proba[actual]
#     help_score = help_proba[actual]

#     performance_tracker["main"].append(main_score)
#     performance_tracker["direction"].append(dir_score)
#     performance_tracker["helper"].append(help_score)

#     # Keep only last 50 trades
#     for key in performance_tracker:
#         if len(performance_tracker[key]) > WINDOW_SIZE:
#             performance_tracker[key] = performance_tracker[key][-WINDOW_SIZE:]


# def compute_weight(scores):
#     if len(scores) == 0:
#         return 1.0  # no history yet

#     weight = 0
#     total = 0

#     for i, val in enumerate(reversed(scores)):
#         w = DECAY ** i
#         weight += val * w
#         total += w

#     return weight / total if total > 0 else 1.0


# def get_dynamic_weights(performance_tracker):

#     w_main = compute_weight(performance_tracker["main"])
#     w_dir  = compute_weight(performance_tracker["direction"])
#     w_help = compute_weight(performance_tracker["helper"])

#     total = w_main + w_dir + w_help

#     if total == 0:
#         return {"main": 1/3, "direction": 1/3, "helper": 1/3}

#     return {
#         "main": w_main / total,
#         "direction": w_dir / total,
#         "helper": w_help / total
#     }



# std_up = np.std(list(probs_up.values()))
# std_down = np.std(list(probs_down.values()))
# agreement_up = 1 - std_up
# agreement_down = 1 - std_down
# ATR_mean_pips = row["ATR_MEAN"] / pip_size
# vol_factor = ATR_pips / ATR_mean_pips
# vol_factor = min(max(vol_factor, 0.8), 1.5)

# base_threshold = 0.7
# threshold_up = base_threshold
# threshold_up -= 0.05 * agreement_up
# threshold_up -= 0.03 * (vol_factor - 1)
# threshold_up = min(max(threshold_up, 0.55), 0.70)

# threshold_down = base_threshold
# threshold_down -= 0.05 * agreement_down
# threshold_down -= 0.03 * (vol_factor - 1)
# threshold_down = min(max(threshold_down, 0.55), 0.70)



# def get_adaptive_threshold_by_direction(history, window=200):
#     best_threshold = 0.7

#     if len(history) < 10:
#         return best_threshold

#     recent = history[-window:]
#     df_hist = pd.DataFrame(recent)
    
#     thresholds = np.arange(0.55, 0.95, 0.02)
    
#     best_score = -np.inf

#     for t in thresholds:
#         subset = df_hist[df_hist["prob"] >= t]
#         if len(subset) < 10:
#             continue
#         win_rate = subset["result"].mean()
#         trades = len(subset)
#         score = win_rate * np.log(trades + 1)
#         if score > best_score:
#             best_score = score
#             best_threshold = t
#     return best_threshold



# IT SEEMS AS IF THE DIRECTION MODEL IS NOT WORKING AS EXPECTED, SO I WANT TO TRAIN DIFFERENT MODEL FOR THRESHOLD CALIBRATIONS FOR EXAMPLE

# # ==========================================================
# # LABEL CREATION (TP/SL LOGIC)
# # ==========================================================
# tp_mult = 1.2
# sl_mult = 0.8

# TARGET_HORIZONS = [1, 2, 3, 4, 6]
# horizon_names = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']

# labels_dict = {name: [] for name in horizon_names}

# for i in range(0, len(df) - max(TARGET_HORIZONS) - 1):
#     high_base = df['High'][i]
#     low_base = df['Low'][i]
#     atr = df['ATR'][i]

#     TP_price = high_base + (tp_mult * atr)
#     SL_price = low_base - (sl_mult * atr)

#     for idx, horizon in enumerate(TARGET_HORIZONS):
#         candle_highs = df['High'][i+1:i+1+horizon].reset_index(drop=True)
#         candle_lows  = df['Low'][i+1:i+1+horizon].reset_index(drop=True)

#     label = None
#     for h in range(horizon):
#         if candle_highs[h] >= TP_price:
#             label = 90 
#             break
#         elif candle_lows[h] <= SL_price:
#             label = 90 
#             break

#     if label is None:
#         max_high = candle_highs.max()
#         min_low  = candle_lows.min()

#         if max_high > high_base + 0.9 * atr:
#             label = 80
#         elif min_low < low_base - 0.9 * atr:
#             label = 80

#         elif max_high > high_base + 0.8 * atr:
#             label = 70
#         elif min_low < low_base - 0.8 * atr:
#             label = 70


#         elif max_high > high_base + 0.7 * atr:
#             label = 60
#         elif min_low < low_base - 0.7 * atr:
#             label = 60

#         elif max_high > high_base + 0.6 * atr:
#             label = 50
#         elif min_low < low_base - 0.6 * atr:
#             label = 50
            
#         elif max_high > high_base + 0.55 * atr:
#             label = 40
#         elif min_low < low_base - 0.55 * atr:
#             label = 40

#         elif max_high > high_base + 0.38 * atr:
#             label = 32
#         elif min_low < low_base - 0.38 * atr:
#             label = 32

#         elif max_high > high_base + 0.2 * atr:
#             label = 20
#         elif min_low < low_base - 0.2 * atr:
#             label = 20

#         elif max_high > high_base + 0.1 * atr:
#             label = 15
#         elif min_low < low_base - 0.1 * atr:
#             label = 15

#         elif max_high > high_base 
#             label = 10
#         elif min_low < low_base 
#             label = 10
#         else:
#             label = 5

# import numpy as np

# thresh_predicts = [np.float64(-25.781016210954732), np.float64(-16.319487752980578), np.float64(-28.516404466841024), np.float64(-26.493378076600777), np.float64(-14.073881897767892)]

# thresh_moves_mean = round(sum(thresh_predicts) / len(thresh_predicts), 2)
# print(thresh_moves_mean)


# a = pd.Series([1,2,3,4,4,4])
# print(a.value_counts())

# def calc_lot_size(balance, risk_percent, sl_pips,pip_value_per_lot, min_lot, max_lot):
#     risk_amount = balance * (risk_percent / 100)
#     lot_cal = risk_amount / (sl_pips * pip_value_per_lot)
#     lot = lot_cal.clip(lower=min_lot,upper=max_lot)
#     return lot


# def normalize_lot(lot, vol_min, vol_max, vol_step):
#     lot = lot.clip(lower=vol_min,upper=vol_max)
#     lot = np.floor(lot / vol_step) * vol_step
#     return round(lot, 2)

# balance = 1000
# risk_percent = 1
# pip_value_per_lot = 10.0
# vol_info = {'min': 0.01, 'max': 500.0, 'step': 0.01}

# df = pd.DataFrame({'a':[12,4,5,8,9,34,5,6]})
# # print(10/df['a'])

# df['LOTS'] = calc_lot_size(balance,risk_percent,df['a'],pip_value_per_lot=pip_value_per_lot,min_lot=0.01,max_lot=2)
# df['MAIN_LOTS'] = normalize_lot(df['LOTS'],vol_info["min"],vol_info["max"],vol_info["step"])
# # print(df)


# a = [{'DIRECTION': 'BUY', 'RESULT': 'WIN', 'WEIGHTED_UP': np.float64(0.9166680361829911), 'THRESH': np.float64(65.62002810292358)}]
# df = pd.DataFrame(a)
# # print(df)

# net_move = 0
# target = np.sign(net_move) * np.log1p(abs(net_move))
# print(target)

# # ==========================================================
# # CLEAN REGRESSION LABEL (PROFESSIONAL VERSION)
# # ==========================================================

# TARGET_HORIZONS = [1, 2, 3, 4, 6]
# horizon_names = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']

# labels_dict = {name: [] for name in horizon_names}
# class_labels_dict = {name+'_CLS': [] for name in horizon_names}

# EPSILON = 1e-6          # noise threshold
# CLIP_VALUE = 3.0        # max ATR-normalized move
# CLASS_THRESHOLD = 0.2   # directional strength threshold

# for i in range(len(df) - max(TARGET_HORIZONS) - 1):

#     entry = df['Close'].iloc[i]
#     atr   = df['ATR'].iloc[i]

#     # Skip bad ATR
#     if atr <= 0 or np.isnan(atr):
#         continue

#     for idx, horizon in enumerate(TARGET_HORIZONS):

#         highs = df['High'].iloc[i+1:i+1+horizon].values
#         lows  = df['Low'].iloc[i+1:i+1+horizon].values

#         max_high = highs.max()
#         min_low  = lows.min()

#         move_up   = (max_high - entry) / atr
#         move_down = (entry - min_low) / atr

#         # =========================
#         # NET MOVE
#         # =========================
#         net_move = move_up - move_down

#         # =========================
#         # STEP 1: REMOVE NOISE
#         # =========================
#         if abs(net_move) < EPSILON:
#             net_move = 0.0

#         # =========================
#         # STEP 2: CLIP EXTREMES
#         # =========================
#         net_move = np.clip(net_move, -CLIP_VALUE, CLIP_VALUE)

#         # =========================
#         # STEP 3: SMOOTH TRANSFORM
#         # =========================
#         net_move = np.tanh(net_move)

#         labels_dict[horizon_names[idx]].append(net_move)

#         # =========================
#         # STEP 4: CLASSIFICATION LABEL
#         # =========================
#         if net_move > CLASS_THRESHOLD:
#             cls = 1      # BUY
#         elif net_move < -CLASS_THRESHOLD:
#             cls = -1     # SELL
#         else:
#             cls = 0      # NO TRADE

#         class_labels_dict[horizon_names[idx]+'_CLS'].append(cls)


# # ALIGN DATAFRAME LENGTH
# df = df.iloc[:len(df) - max(TARGET_HORIZONS) - 1].copy()

# # ASSIGN LABELS
# for name in horizon_names:
#     df[name] = labels_dict[name]
#     df[name+'_CLS'] = class_labels_dict[name+'_CLS']


# # ==========================================================
# # DEBUG OUTPUT
# # ==========================================================

# print("\n📊 CLEANED REGRESSION DISTRIBUTION:")
# for t in horizon_names:
#     print(f"\n{t}")
#     print(df[t].describe())

# print("\n📊 CLASS DISTRIBUTION:")
# for t in horizon_names:
#     print(f"\n{t}_CLS")
#     print(df[t+'_CLS'].value_counts(normalize=True))


    

# N_SPLITS = 5
# N_TRIALS = 30

# # ==========================================================
# # EVALUATION FUNCTION (TIME SERIES SAFE)
# # ==========================================================


# def evaluate_model(X, y, params):

#     tscv = TimeSeriesSplit(n_splits=N_SPLITS)
#     scores = []

#     for train_idx, val_idx in tscv.split(X):

#         X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

#         # ✅ Weight only training data
#         # sample_weight = 1 + (np.abs(y_train) / 100)
#         # sample_weight = np.log1p(abs(y_train))
#         sample_weight = np.log1p(abs(y_train)) ** 1.5

#         model = LGBMRegressor(**params)

#         model.fit(
#             X_train,
#             y_train,
#             sample_weight=sample_weight,

#             eval_set=[(X_valid, y_valid)],
#             eval_metric="l1",  # MAE

#             callbacks=[
#                 early_stopping(stopping_rounds=50),
#                 log_evaluation(0)  # silent
#             ])

#         preds = model.predict(X_valid, num_iteration=model.best_iteration_)

#         mae = mean_absolute_error(y_valid, preds)
#         scores.append(mae)

#     return np.mean(scores)

# # ==========================================================
# # TRAIN LOOP (FULL FEATURES)
# # ==========================================================
# for target in horizon_names:

#     print(f"\n==============================")
#     print(f"🚀 REGRESSION TRAINING: {target}")
#     print(f"==============================")

#     # Prevent leakage
#     drop_targets = [t for t in horizon_names if t != target]
#     df_train_target = df.drop(columns=drop_targets)

#     X = df_train_target.drop(columns=[target])
#     y = df_train_target[target]

#     # ======================================================
#     # OPTUNA TUNING
#     # ======================================================
#     def objective(trial):

#         params = {
#             "objective": "huber",   # ✅ CHANGE HERE
#             "metric": "mae",        # keep this for evaluation consistency

#             "alpha": trial.suggest_float("alpha", 0.85, 0.95),  # ✅ NEW (VERY IMPORTANT)

#             "n_estimators": trial.suggest_int("n_estimators", 200, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 20, 300),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
#             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

#             "random_state": 42,
#             "n_jobs": -1
#         }

#         return evaluate_model(X, y, params)

#     study = optuna.create_study(direction="minimize")  # MAE → minimize
#     study.optimize(objective, n_trials=N_TRIALS)

#     print("🏆 BEST MAE:", study.best_value)
#     print("🏆 BEST PARAMS:", study.best_params)
    
    
#     # ======================================================
#     #  TRAIN FINAL MODEL
#     #  ======================================================

#     final_model = LGBMRegressor(
#         **study.best_params,
#         objective="regression",
#         random_state=42,
#         n_jobs=-1
#     )

#     split = int(len(X) * 0.9)

#     X_train, X_val = X.iloc[:split], X.iloc[split:]
#     y_train, y_val = y.iloc[:split], y.iloc[split:]

#     # sample_weight = 1 + (np.abs(y_train) / 100)
#     # sample_weight = np.log1p(abs(y_train))
#     sample_weight = np.log1p(abs(y_train)) ** 1.5

#     final_model.fit(
#         X_train,
#         y_train,
#         sample_weight=sample_weight,
#         eval_set=[(X_val, y_val)],
#         eval_metric="l1",
#         callbacks=[early_stopping(50), log_evaluation(0)]
#     )

#     # Safe best iteration
#     best_iter = final_model.best_iteration_
#     if best_iter is None:
#         best_iter = final_model.n_estimators

#     # Validation performance
#     val_preds = final_model.predict(X_val, num_iteration=best_iter)
#     val_mae = mean_absolute_error(y_val, val_preds)

#     # Save
#     joblib.dump(
#         {
#             "model": final_model,
#             "features": X.columns.tolist(),
#             "target": target,
#             "params": study.best_params,
#             "best_iteration": best_iter,
#             "val_mae": val_mae
#         },
#         f"/workspaces/PRIVATE-FOREX-BOT/FOREX GITHUB/ALL_MODELS/HL_THRESH_{target}_{SYMBOL}_model.pkl"
#     )

#     print(f"💾 REGRESSION MODEL SAVED: {target} | MAE: {val_mae:.5f}")




# import ta
# import shap
# import joblib
# import optuna
# import numpy as np
# import pandas as pd
# from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# from sklearn.model_selection import TimeSeriesSplit
# from func import apply_features, SYMBOL
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import roc_auc_score

# # ==========================================================
# # LOAD DATA
# # ==========================================================
# data = pd.read_csv(f'/workspaces/PRIVATE-FOREX-BOT/FOREX GITHUB/CSV_FILES/MT5_5M_{SYMBOL}_Exchange_Rate_Dataset.csv') 

# df = apply_features(data)
# print('after APPLY FEATURES :', len(df))

# df.dropna(inplace=True)
# print('after apply dropna:', len(df))

# df.reset_index(drop=True, inplace=True)
# print('COLUMNS:', len(df.columns))
# print('ROWS:', len(df))


# # ==========================================================
# # PARAMETERS
# # ==========================================================
# tp_mult = 1.5
# sl_mult = 1.0

# TARGET_HORIZONS = [1, 2, 3, 4, 6]
# horizon_names = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']

# labels_dict = {name: [] for name in horizon_names}

# # ==========================================================
# # STEP 1: COLLECT FULL MARKET DRAWDOWNS (NO BIAS)
# # ==========================================================
# drawdowns = []

# for i in range(len(df) - max(TARGET_HORIZONS) - 1):

#     entry = df['Close'].iloc[i]
#     atr = df['ATR'].iloc[i]

#     for idx, horizon in enumerate(TARGET_HORIZONS):

#         highs = df['High'].iloc[i+1:i+1+horizon].values
#         lows  = df['Low'].iloc[i+1:i+1+horizon].values

#         # worst movement in BOTH directions (important fix)
#         adverse_up   = highs.max() - entry
#         adverse_down = entry - lows.min()

#         worst_move = max(adverse_up, adverse_down)

#         drawdowns.append(worst_move / atr)

# # robust buffer (market noise threshold)
# buffer_ratio = np.percentile(drawdowns, 80)

# # ==========================================================
# # STEP 2: FINAL LABEL GENERATION (SINGLE PASS, CLEAN LOGIC)
# # ==========================================================
# for i in range(len(df) - max(TARGET_HORIZONS) - 1):

#     entry = df['Close'].iloc[i]
#     atr = df['ATR'].iloc[i]

#     TP = entry + tp_mult * atr
#     SL = entry - sl_mult * atr

#     buffer = 0.5 * buffer_ratio * atr

#     for idx, horizon in enumerate(TARGET_HORIZONS):

#         highs = df['High'].iloc[i+1:i+1+horizon].values
#         lows  = df['Low'].iloc[i+1:i+1+horizon].values

#         label = -1

#         for h in range(horizon):

#             # ===== BUY WIN CONDITION =====
#             if highs[h] > TP + buffer:
#                 label = 1
#                 break

#             # ===== SELL WIN CONDITION =====
#             if lows[h] < SL - buffer:
#                 label = 0
#                 break

#         labels_dict[horizon_names[idx]].append(label)


# # Trim df to match label length
# df = df.iloc[:len(df) - max(TARGET_HORIZONS) - 1].copy()

# for name in horizon_names:
#     df[name] = labels_dict[name]

# print("Label Distribution:")
# for t in horizon_names:
#     print(t, df[t].value_counts(normalize=True))



# # ==========================================================
# # SETTINGS
# # ==========================================================
# N_SPLITS = 5
# N_TRIALS = 30
# research = []

# # ==========================================================
# # EVALUATION FUNCTION (TIME SERIES SAFE)
# # ==========================================================
# def evaluate_model(X, y, params):

#     tscv = TimeSeriesSplit(n_splits=N_SPLITS)
#     scores = []

#     for train_idx, val_idx in tscv.split(X):

#         X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

#         model = LGBMClassifier(**params)

#         model.fit(
#             X_train,
#             y_train,

#             eval_set=[(X_valid, y_valid)],
#             eval_metric="auc",

#             callbacks=[
#                 early_stopping(stopping_rounds=50),
#                 log_evaluation(0)  # silent
#             ])

#         preds = model.predict_proba(X_valid,num_iteration=model.best_iteration_)[:, 1]

#         auc = roc_auc_score(y_valid, preds)
#         scores.append(auc)
#         research.append(f'initial_auc{scores}')

#     return np.mean(scores)


# # ==========================================================
# # TRAIN LOOP (NO FEATURE SELECTION)
# # ==========================================================
# for target in horizon_names:

#     print(f"\n==============================")
#     print(f"🚀 FULL FEATURE TRAINING: {target}")
#     print(f"==============================")

#     # Prevent leakage
#     drop_targets = [t for t in horizon_names if t != target]
#     df_train_target = df.drop(columns=drop_targets)
#     df_train_target = df_train_target[df_train_target[target] != -1].copy()
#     df_train_target.reset_index(drop=True, inplace=True)

#     X = df_train_target.drop(columns=[target])
#     y = df_train_target[target]

#     # Handle class imbalance
#     classes = np.unique(y)
#     class_weights = compute_class_weight(
#         class_weight="balanced",
#         classes=classes,
#         y=y
#     )
#     class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
#     print(y.value_counts(normalize=True))

#     print("Class Weights:", class_weight_dict)

#     # ======================================================
#     # OPTUNA TUNING (FULL FEATURES)
#     # ======================================================
#     def objective(trial):

#         params = {
#             "objective": "binary",
#             "metric": "auc",
#             "n_estimators": trial.suggest_int("n_estimators", 200, 800),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
#             "num_leaves": trial.suggest_int("num_leaves", 20, 300),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
#             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#             "random_state": 42,
#             "n_jobs": -1,
#             "class_weight": class_weight_dict
#         }

#         return evaluate_model(X, y, params)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=N_TRIALS)

#     print("🏆 BEST AUC:", study.best_value)
#     print("🏆 BEST PARAMS:", study.best_params)

#     # ======================================================
#     # TRAIN FINAL MODEL
#     # ======================================================
#     final_model = LGBMClassifier(
#         **study.best_params,
#         objective="binary",
#         random_state=42,
#         n_jobs=-1,
#         class_weight=class_weight_dict
#     )

#     split = int(len(X) * 0.9)

#     X_train, X_val = X.iloc[:split], X.iloc[split:]
#     y_train, y_val = y.iloc[:split], y.iloc[split:]

#     final_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric="auc",
#         callbacks=[early_stopping(50), log_evaluation(0)]
#     )

#     # Validation score
#     val_preds = final_model.predict_proba(
#         X_val, num_iteration=final_model.best_iteration_)[:, 1]
#     val_auc = roc_auc_score(y_val, val_preds)

#     # Save
#     joblib.dump(
#         {
#             "model": final_model,
#             "features": X.columns.tolist(),
#             "target": target,
#             "params": study.best_params,
#             "best_iteration": final_model.best_iteration_,
#             "val_auc": val_auc,
#             "feature_importance": final_model.feature_importances_.tolist()
#         },
#         f"/workspaces/PRIVATE-FOREX-BOT/FOREX GITHUB/ALL_MODELS/HL_HELPER_{target}_{SYMBOL}_model.pkl"
#     )

#     print(f"💾 FULL MODEL SAVED: {target} | AUC: {val_auc:.4f}")
#     print(research)

# HIGH_TARGETS = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']
# weights = {
# 'THL_5M'  : 0.6,
# 'THL_10M' : 1.5,
# 'THL_15M' : 1.6,
# 'THL_20M' : 1.2,
# 'THL_30M' : 1.0
# }
# weight_list = [weights[t] for t in HIGH_TARGETS]
# total_weight = sum(weight_list)

# # print(weight_list)
# # print(total_weight)



# net_move = -100
# net_move = np.sign(net_move) * np.log1p(abs(net_move))
# print(net_move)

# a = [1,3.06322,4]
# b = [2,4,5]
# print(round(a*100,2))



# import pandas as pd

# # =========================
# # 1. LOAD DATA
# # =========================
# # file_path = 'CSV_FILES/test.csv'
# file_path = 'CSV_FILES/EURUSD_M5_201901020600_202512312355.csv'

# data = pd.read_csv(
#     file_path,
#     sep='\t'
# )

# # =========================
# # 2. CLEAN COLUMN NAMES
# # =========================
# data.columns = data.columns.str.strip()

# # =========================
# # 3. CREATE DATETIME COLUMN
# # =========================
# data['Date'] = pd.to_datetime(data['DATE'] + ' ' + data['TIME'])

# # =========================
# # 4. RENAME COLUMNS
# # =========================
# data = data.rename(columns={
#     'OPEN': 'Open',
#     'HIGH': 'High',
#     'LOW': 'Low',
#     'CLOSE': 'Close',
#     'TICKVOL': 'Volume',   # ✅ Use TickVolume as Volume
#     'SPREAD': 'Spread'
# })

# # =========================
# # 5. DROP UNUSED COLUMN
# # =========================
# data = data.drop(columns=['VOL'])   # ❌ Remove useless volume column

# # =========================
# # 6. SELECT & ORDER COLUMNS
# # =========================
# data = data[
#     ['Date', 'Open', 'High', 'Low', 'Close',
#      'Volume', 'Spread']
# ]

# # =========================
# # 7. CLEAN DATA
# # =========================
# data = data.sort_values('Date')
# data = data.drop_duplicates()
# data = data.reset_index(drop=True)

# # =========================
# # 8. SAVE CLEAN DATASET
# # =========================
# output_file = 'MT5_Cleaned_EURUSD_M5_FINAL.csv'

# data.to_csv(output_file, index=False)

# print("✅ Clean dataset created successfully!")
# print(f"📁 Saved as: {output_file}")

# # Preview
# print("\n🔍 Preview:")
# print(data.head())




# import os
# import time
# import joblib
# import pandas as pd
# import MetaTrader5 as mt5
# from func import apply_features,calc_lot_size,place_buy,place_sell,move_sl_and_partial_close,SYMBOL,get_pip_info,log_trade,wait_for_new_candle,Entry_Filtering


# # Initialize MT5 once
# if not mt5.initialize():
#     raise RuntimeError("❌ MT5 initialization failed")
# print("✅ MT5 initialized successfully")
 

# HIGH_TARGETS = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']

# BASE_PATH = "ALL_MODELS"

# def load_models(model_type, symbol):
#     models_dict = {}

#     for target in HIGH_TARGETS:
#         file_path = f"{BASE_PATH}/HL_{model_type}_{target}_{symbol}_model.pkl"

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Model not found: {file_path}")

#         bundle = joblib.load(file_path)

#         models_dict[target] = {
#             "model": bundle["model"],
#             "features": bundle["features"]
#         }

#         print(f"✅ Loaded {model_type} model for {target}")

#     return models_dict


# # LOAD ALL MODELS HERE
# MAIN_MODELS = load_models("MAIN", SYMBOL)
# HELPER_MODELS = load_models("HELPER", SYMBOL)
# # DIRECTION_MODELS = load_models("DIRECTION", SYMBOL)
# THRESH_MODELS = load_models("THRESH", SYMBOL)

# weights = {
# 'THL_5M'  : 1.5,
# 'THL_10M' : 1.4,
# 'THL_15M' : 1.1,
# 'THL_20M' : 1.0,
# 'THL_30M' : 0.7
# }

# model_weights = {
#     "main": 0.6,       # most reliable
#     "helper": 0.4      # strict confirmation
# }

# delay_time = 60*2
# risk_percent = 1

# W_threshold = 0.535 # DIRECTION 
# threshold = 0.0 # THRESHOLD (+/-)


# PREV_PRED_PROB = ''

# try:
#     while True:
#         TIMEFRAME = mt5.TIMEFRAME_M5
#         N_BARS = 350
#         wait_for_new_candle(mt5,SYMBOL, TIMEFRAME)

#         rates = mt5.copy_rates_from_pos(SYMBOL,TIMEFRAME,1,N_BARS )

#         if rates is None or len(rates) < N_BARS:
#             mt5.shutdown()
#             raise RuntimeError("❌ Failed to fetch enough closed candles")

#         data = pd.DataFrame(rates)
#         data['Date'] = pd.to_datetime(data['time'], unit='s')

#         data.rename(columns={
#             'open': 'Open',
#             'high': 'High',
#             'low': 'Low',
#             'close': 'Close',
#             'tick_volume': 'Volume'
#         }, inplace=True)

#         new_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
#         new_df.sort_values('Date', inplace=True)
#         new_df.reset_index(drop=True, inplace=True)
#         # assert len(df) == N_BARS
#         # assert df.isnull().sum().sum() == 0
#         # mt5.shutdown()
#         print(new_df.tail())

#         df = apply_features(new_df)
#         df.dropna(inplace=True)
#         df.reset_index(drop=True, inplace=True)

#         up_moves = {}
#         down_moves = {}
#         thresh_predicts = []
#         for target in HIGH_TARGETS:

#             print(f'\n [[ CURRENTLY PREDICTING TARGET : {target} ]]')

#             main_model = MAIN_MODELS[target]["model"]
#             help_model = HELPER_MODELS[target]["model"]
#             thresh_model = THRESH_MODELS[target]["model"]

#             main_cols = MAIN_MODELS[target]["features"]
#             help_cols = HELPER_MODELS[target]["features"]
#             thresh_cols  = THRESH_MODELS[target]["features"]
            
#             current_candle = df.tail(1)

#             X_main = current_candle[main_cols]
#             X_help = current_candle[help_cols]
#             X_thresh = current_candle[thresh_cols]

#             # Safety check
#             assert set(main_cols) == set(X_main.columns)
#             assert set(help_cols) == set(X_help.columns)        
#             assert set(thresh_cols) == set(X_thresh.columns)

#             main_proba = main_model.predict_proba(X_main)[0]
#             help_proba = help_model.predict_proba(X_help)[0]
#             thres_proba = thresh_model.predict(X_thresh)[0]

#             thresh_predicts.append(thres_proba)

#             up_prob = (
#                 main_proba[1] * model_weights["main"] +
#                 help_proba[1] * model_weights["helper"]
#             )

#             down_prob = (
#                 main_proba[0] * model_weights["main"] +
#                 help_proba[0] * model_weights["helper"]
#             )

#             # Normalize (optional but safer)
#             total = up_prob + down_prob
#             up_prob /= total
#             down_prob /= total

#             up_moves[target] = round(up_prob * 100, 2)
#             down_moves[target] = round(down_prob * 100, 2)

#             direction = "UP" if up_prob > down_prob else "DOWN"
#             print(f"MAIN  : {main_proba}")
#             # print(f"DIR   : {dir_proba}")
#             print(f"HELP  : {help_proba}")
#             print(f"THRESH  : {thres_proba}")
#             print(f"➡ FINAL → {direction} ({round(max(up_prob, down_prob)*100,2)}%)")


#         up_moves_mean = round(sum(up_moves.values()) / len(up_moves), 2)
#         down_moves_mean = round(sum(down_moves.values()) / len(down_moves), 2)

#         thresh_moves_mean = round(sum(thresh_predicts) / len(thresh_predicts), 2)

#         probs_up = {k: v/100 for k, v in up_moves.items()}
#         probs_down = {k: v/100 for k, v in down_moves.items()}

#         print('===================== ACCOUNT INFORMATIONS ==============================')
#         account_info = mt5.account_info()
#         balance = account_info.balance
#         print("Account Number:", account_info.login)
#         print("Balance:", account_info.balance)
#         print("Equity:", account_info.equity)
#         print("Free Margin:", account_info.margin_free)
#         print("Leverage:", account_info.leverage)  

#         pip_info = get_pip_info(mt5, SYMBOL)
#         pip_size = pip_info["pip_size"]
#         pip_value_per_lot = pip_info["pip_value_per_lot"]

#         # Get current tick
#         tick = mt5.symbol_info_tick(SYMBOL)
#         ask_price = tick.ask
#         bid_price = tick.bid
#         spread = ask_price - bid_price  # real-time spread

#         symbol_info = mt5.symbol_info(SYMBOL)
#         spread_points = spread / symbol_info.point
#         max_spread_points = 30   # example threshold

#         if spread_points > max_spread_points:
#             print(f"⚠️ Spread too high: {spread_points:.2f} points, skipping trade")
#             time.sleep(delay_time)
#             continue

#         print('pip_size:',pip_size)
#         print('pip_value_per_lot:',pip_value_per_lot)
#         print('ask_price:',ask_price)
#         print('bid_price:',bid_price)
#         print('spread:',spread)

#         row = df.iloc[-1]
#         # ATR_pips = row["ATR"] / pip_size

#         # # Apply ATR multiplier and enforce min/max limits
#         # SL_pips = ATR_pips * 1.0
#         # TP_pips = ATR_pips * 1.3

#         # # BUY trade SL/TP
#         # entry_buy = ask_price
#         # SL_buy = (entry_buy - (SL_pips * pip_size))
#         # TP_buy = (entry_buy + (TP_pips * pip_size))

#         # # SELL trade SL/TP
#         # entry_sell = bid_price
#         # SL_sell = (entry_sell + (SL_pips * pip_size))
#         # TP_sell = (entry_sell - (TP_pips * pip_size))

#         print('\n=======================================================================')

#         positions = mt5.positions_get(symbol=SYMBOL)

#         if positions is None:
#             print("⚠️ Error fetching positions")
#             has_open_trade = False
#             has_buy = False
#             has_sell = False

#         elif len(positions) == 0:
#             has_open_trade = False
#             has_buy = False
#             has_sell = False

#         else:
#             buy_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
#             sell_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

#             has_buy = len(buy_positions) > 0
#             has_sell = len(sell_positions) > 0
#             has_open_trade = has_buy or has_sell   # ✅ FIX

#             if has_buy:
#                 print("⚠️ BUY exists")

#             if has_sell:
#                 print("⚠️ SELL exists")

                
#         total_weight = sum(weights.values())
#         weighted_up = sum(probs_up[k] * weights[k] for k in weights) / total_weight
#         weighted_down = sum(probs_down[k] * weights[k] for k in weights) / total_weight

#         if PREV_PRED_PROB == weighted_up or PREV_PRED_PROB == weighted_down:
#             print('\n CURRENT CANDLE PREDICTION SAME AS BEFORE')
#             time.sleep(delay_time)
#             continue
#         print('\n================ WEIGHTED RESULT =================')
#         print('THRESHOLD MEAN :',thresh_moves_mean)
#         print(f'WEIGHTED UP   : {round(weighted_up,2)}%')
#         print(f'WEIGHTED DOWN : {round(weighted_down,2)}%')

#         if not has_open_trade:
#             ##================================= MAIN DIRECTION TRADING ==================================================
#             if weighted_up >= W_threshold:
#                 print('CHECKING FOR BUY ENTRY LOGIC')
#                 entry_buy = Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
#                     direction="BUY",atr_value=row["ATR"])

#                 if entry_buy is None:
#                     print("❌ BUY skipped (no confirmation)")
#                 else:
#                     PREV_PRED_PROB = weighted_up
#                     print("🚀 FINAL SIGNAL: BUY")

#                     atr = row["ATR"]
#                     SL_distance = atr * 1.0
#                     TP_distance = atr * 1.3
#                     SL_pips = SL_distance/ pip_size 

#                     SL_buy = entry_buy - SL_distance
#                     TP_buy = entry_buy + TP_distance
#                     lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
                    
#                     print(f'LOTS SIZE USED: {lot_size}')

#                     result = place_buy(mt5, SYMBOL, lot_size, entry_buy, SL_buy, TP_buy)
#                     print("BUY ORDER RESULT:", result)
#                     log_trade(mt5=mt5,symbol=SYMBOL,direction="BUY",entry_price=entry_buy,SL=SL_buy,TP=TP_buy,lot_size=lot_size,
#                         proba_up=weighted_up,proba_down=weighted_down,order_result=result)

#             elif weighted_down >= W_threshold:
#                 print('CHECKING FOR SELL ENTRY LOGIC')
#                 entry_sell = Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
#                     direction="SELL",atr_value=row["ATR"])

#                 if entry_sell is None:
#                     print("❌ SELL skipped (no confirmation)")
#                 else:
#                     PREV_PRED_PROB = weighted_down
#                     print("🔻 FINAL SIGNAL: SELL")

#                     atr = row["ATR"]
#                     SL_distance = atr * 1.0
#                     TP_distance = atr * 1.3
#                     SL_pips = SL_distance/ pip_size 

#                     SL_sell = entry_sell + SL_distance
#                     TP_sell = entry_sell - TP_distance
#                     lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
                    
#                     print(f'LOTS SIZE USED: {lot_size}')

#                     result = place_sell(mt5, SYMBOL, lot_size, entry_sell, SL_sell, TP_sell)
#                     print("SELL ORDER RESULT:", result)
#                     log_trade(mt5=mt5,symbol=SYMBOL,direction="SELL",entry_price=entry_sell,
#                         SL=SL_sell,TP=TP_sell,lot_size=lot_size,proba_up=weighted_up,proba_down=weighted_down,order_result=result)
                
#             else:
#                 print("⏳ FINAL SIGNAL: NO TRADE")

#             while True:
#                 print('CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION')
#                 output = move_sl_and_partial_close(mt5 = mt5, SYMBOL = SYMBOL, atr_value = atr)
#                 if output == 1:
#                     print('TP/SL + PARTIAL CONDITION MET')
#                     break
                
#         elif has_open_trade:
#             while True:
#                 print('CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION')
#                 output = move_sl_and_partial_close(mt5 = mt5, SYMBOL = SYMBOL, atr_value = atr)
#                 if output == 1:
#                     print('TP/SL + PARTIAL CONDITION MET')
#                     break


# finally:
#     # Shutdown MT5 only once, when the bot stops
#     mt5.shutdown()
#     print("✅ MT5 shutdown successfully")




    # # =========================
    # # STEP 1: BASIC CHECKS
    # # =========================
    # print("\n🧪 BASIC DATA CHECK")

    # print("Shape:", df_train_target.shape)
    # print("NaN count:", df_train_target.isna().sum().sum())
    # print("Inf count:", np.isinf(df_train_target.select_dtypes(include=[np.number])).sum().sum())

    # # =========================
    # # STEP 2: LABEL CHECK
    # # =========================

    # print("\n🧪 LABEL CHECK")
    # print("Unique labels:", np.unique(y))
    # print("Distribution:")
    # print(y.value_counts(normalize=True))

    # # Check if both classes exist
    # if len(np.unique(y)) < 2:
    #     print("❌ ERROR: Only one class present! Skipping...")
    #     continue

    # # =========================
    # # STEP 3: FEATURE CHECK
    # # =========================

    # print("\n🧪 FEATURE CHECK")

    # print("Feature count:", X.shape[1])

    # # Constant features
    # constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    # print("Constant features:", len(constant_cols))

    # if constant_cols:
    #     print("⚠ Constant columns:", constant_cols)

    # # Extreme values
    # print("\nMax values:")
    # print(X.max().sort_values(ascending=False).head())

    # print("\nMin values:")
    # print(X.min().sort_values().head())

    # # =========================
    # # STEP 4: CLASS WEIGHTS
    # # =========================
    # classes = np.unique(y)
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=classes,
    #     y=y
    # )

    # class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

    # print("\n🧪 CLASS WEIGHT CHECK")
    # for k, v in class_weight_dict.items():
    #     print(f"Class {k}: Weight {v}")

    # # =========================
    # # STEP 5: SANITY SAMPLE
    # # =========================
    # print("\n🧪 LABEL SAMPLE")
    # print(df_train_target[[target]].sample(10, random_state=42))

    # # =========================
    # # STEP 6: FINAL ASSERTIONS
    # # =========================
    # assert df_train_target.isna().sum().sum() == 0, "❌ NaNs detected!"
    # assert len(X) == len(y), "❌ X and y misaligned!"
    # assert set(np.unique(y)).issubset({0, 1}), "❌ Invalid labels detected!"

    # print("\n✅ DATA READY FOR TRAINING")




# THL_5M THL_5M
# -1    0.854467
#  0    0.111747
#  1    0.033785
# Name: proportion, dtype: float64
# THL_10M THL_10M
# -1    0.669000
#  0    0.234469
#  1    0.096531
# Name: proportion, dtype: float64
# THL_15M THL_15M
# -1    0.522082
#  0    0.321232
#  1    0.156686
# Name: proportion, dtype: float64
# THL_20M THL_20M
# -1    0.412026
#  0    0.382362
#  1    0.205613
# Name: proportion, dtype: float64
# THL_30M THL_30M
#  0    0.460864
#  1    0.273332
# -1    0.265804
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_5M
# ==============================
# THL_5M
# 0    0.76785
# 1    0.23215
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.6511688418400187), np.int64(1): np.float64(2.15377995198623)}
# LEN X AFTER PREPROCESSING :  95098
# ==============================
# 🚀 FULL FEATURE TRAINING: THL_20M
# ==============================
# THL_20M
# 0    0.650303
# 1    0.349697
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.7688720967929142), np.int64(1): np.float64(1.4298101327061485)}
# LEN X AFTER PREPROCESSING :  384210



# import os
# import ta
# import joblib
# import numpy as np
# import pandas as pd
# from scipy.stats import entropy
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import MetaTrader5 as mt5



# SYMBOL = "EURUSD"    # EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD
# _CANDLE = '5M'

# def apply_features(df):
#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])
#     df = df.sort_values("Date").reset_index(drop=True)
#     df["Close"] = pd.to_numeric(df["Close"])
#     df["High"] = pd.to_numeric(df["High"])
#     df["Low"] = pd.to_numeric(df["Low"])
#     df["Open"] = pd.to_numeric(df["Open"])
#     df["Volume"] = pd.to_numeric(df["Volume"])
#     df['Hour'] = df['Date'].dt.hour
#     df['london_session'] = ((df['Hour'] >= 7) & (df['Hour'] <= 16)).astype(int)
#     df['ny_session'] = ((df['Hour'] >= 13) & (df['Hour'] <= 22)).astype(int)
#     df['asia_session'] = ((df['Hour'] >= 0) & (df['Hour'] <= 8)).astype(int)

#     df['Weekday'] = df['Date'].dt.weekday
#     df["Date_ordinal"] = df["Date"].apply(lambda x: x.toordinal())
#     df['hour_of_day'] = df['Date'].dt.hour
#     df['day_of_week'] = df['Date'].dt.weekday

#     df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=8)
#     df["ATR_3"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=3)
#     df["ATR_6"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=6)
#     df["ATR_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
#     df["ATR_20"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=20)
#     df["ATR_36"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=36)
#     df["ATR_40"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=40)

#     df['ADX6'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=6)
#     df['ADX8'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=8)
#     df['ADX14'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
#     df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=12)
#     bb = ta.volatility.BollingerBands(df["Close"], window=10)
#     df["BB_H"] = bb.bollinger_hband()
#     df["BB_L"] = bb.bollinger_lband()
#     df["bollinger_width"] = df["BB_H"] - df["BB_L"]
#     df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-6)
#     df["RSI"] = ta.momentum.rsi(df["Close"], window=7)
#     df["STOCH"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
#     df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])
#     df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
#     df['EMA7'] = ta.trend.ema_indicator(df['Close'], window=7)
#     df['EMA10'] = ta.trend.ema_indicator(df['Close'], window=10)
#     df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
#     df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
#     df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)
#     df['TREND'] = np.where(df['EMA50'] > df['EMA200'], 1, -1)
#     df['trend_strength'] = (df['EMA50'] - df['EMA200']) / df['Close']
#     df['SMA5'] = ta.trend.sma_indicator(df['Close'], window=5)
#     df['SMA7'] = ta.trend.sma_indicator(df['Close'], window=7)
#     df['SMA10'] = ta.trend.sma_indicator(df['Close'], window=10)
#     df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
#     df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
#     df['SMA100'] = ta.trend.sma_indicator(df['Close'], window=100)
#     df["WMA_5"] = ta.trend.wma_indicator(df["Close"], window=5)
#     df["WMA_7"] = ta.trend.wma_indicator(df["Close"], window=7)
#     df["WMA_10"] = ta.trend.wma_indicator(df["Close"], window=10)
#     df["WMA_20"] = ta.trend.wma_indicator(df["Close"], window=20)
#     df["WMA_30"] = ta.trend.wma_indicator(df["Close"], window=30)
#     df["WMA_50"] = ta.trend.wma_indicator(df["Close"], window=50)
#     df["MACD"] = ta.trend.macd(df["Close"])
#     df["MACD_signal"] = ta.trend.macd_signal(df["Close"])
#     df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
#     df["+DI"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"])
#     df["-DI"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"])
#     df["Ichimoku_A"] = ta.trend.ichimoku_a(df["High"], df["Low"])
#     df["Ichimoku_B"] = ta.trend.ichimoku_b(df["High"], df["Low"])
#     df["TRIX"] = ta.trend.trix(df["Close"])
#     df["KST"] = ta.trend.kst(df["Close"])
#     df["Stoch_RSI"] = ta.momentum.stochrsi(df["Close"])
#     # df["CCI"] = ta.momentum.cci(df["High"], df["Low"], df["Close"])
#     df["ROC"] = ta.momentum.roc(df["Close"])
#     df["AO"] = ta.momentum.awesome_oscillator(df["High"], df["Low"])
#     df["Donchian_high"] = ta.volatility.donchian_channel_hband(df["High"], df["Low"], df["Close"])
#     df["Donchian_low"] = ta.volatility.donchian_channel_lband(df["High"], df["Low"], df["Close"])
#     df["Keltner_high"] = ta.volatility.keltner_channel_hband(df["High"], df["Low"], df["Close"])
#     df["Keltner_low"] = ta.volatility.keltner_channel_lband(df["High"], df["Low"], df["Close"])
#     df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
#     df["MFI"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"])
#     ema = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
#     ema_of_ema = ta.trend.EMAIndicator(close=ema, window=20).ema_indicator()
#     df['DEMA_20'] = 2 * ema - ema_of_ema
#     ema2 = ema_of_ema
#     ema3 = ta.trend.EMAIndicator(close=ema2, window=20).ema_indicator()
#     df['TEMA_20'] = 3 * ema - 3 * ema2 + ema3
#     df['KAMA_5'] = ta.momentum.KAMAIndicator(close=df['Close'], window=5).kama()
#     df['KAMA_7'] = ta.momentum.KAMAIndicator(close=df['Close'], window=7).kama()
#     df['KAMA_15'] = ta.momentum.KAMAIndicator(close=df['Close'], window=15).kama()
#     df['KAMA_20'] = ta.momentum.KAMAIndicator(close=df['Close'], window=20).kama()
#     df['KAMA_30'] = ta.momentum.KAMAIndicator(close=df['Close'], window=30).kama()
#     df['KAMA_40'] = ta.momentum.KAMAIndicator(close=df['Close'], window=40).kama()
#     vol_change = df['Volume'].pct_change()
#     df['Volume_change_3']  = vol_change.rolling(3).mean()
#     df['Volume_change_10'] = vol_change.rolling(10).mean()
#     df['Volume_change_20'] = vol_change.rolling(20).mean()
#     df['Volume_change_30'] = vol_change.rolling(30).mean()
#     df['Volume_change_40'] = vol_change.rolling(40).mean()
#     df["AD"] = ta.volume.acc_dist_index(df["High"], df["Low"], df["Close"], df["Volume"])

#     df['ATR_MEAN'] = df['ATR'].rolling(10).mean()
#     df['ATR_MEAN_20'] = df['ATR'].rolling(20).mean()
#     df['ATR_MEAN_30'] = df['ATR'].rolling(30).mean()
#     df['ATR_MEAN_40'] = df['ATR'].rolling(40).mean()
#     df['ATR_MEAN_50'] = df['ATR'].rolling(50).mean()
#     df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
#     df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
#     df['pivot_high'] = ((df['High'] > df['High'].shift(1)) &(df['High'] > df['High'].shift(2))).shift(2).fillna(0).astype(int)
#     df['pivot_low'] = (((df['Low'] < df['Low'].shift(1)) &df['Low'] < df['Low'].shift(2))).shift(2).fillna(0).astype(int)
#     df['dist_to_prev_high'] = (df['Close'] - df['High'].rolling(5).max()) / df['ATR']
#     df['dist_to_prev_low'] = (df['Close'] - df['Low'].rolling(5).min()) / df['ATR']
#     df['slope_ema5']  = (df['EMA5'] - df['EMA5'].shift(1)) / df['Close']
#     df['slope_ema7']  = (df['EMA7'] - df['EMA7'].shift(1)) / df['Close']
#     df['slope_ema10'] = (df['EMA10'] - df['EMA10'].shift(1)) / df['Close']
#     df['ema5_10_state'] = (df['EMA5'] > df['EMA10']).astype(int)
#     df['ema10_20_state'] = (df['EMA10'] > df['EMA20']).astype(int)
#     df['ema5_20_state'] = (df['EMA5'] > df['EMA20']).astype(int)
#     df['ema5_10_cross_event'] = ((df['EMA5'] > df['EMA10']) &(df['EMA5'].shift(1) <= df['EMA10'].shift(1))).astype(int)
#     df['ma_alignment'] = ((df['EMA5'] > df['EMA10']) &(df['EMA10'] > df['EMA20'])).astype(int)
#     df['price_diff'] = (df['Close'] - df['Close'].shift(1)) / df['Close']
#     df['rsi_diff'] = df['RSI'] - df['RSI'].shift(1)
#     df['Dist_from_EMA200'] = (df['Close'] - df['EMA200']) / df['ATR']
#     df['divergence_flag'] = (
#         (df['price_diff'].rolling(3).mean() > 0) &
#         (df['rsi_diff'].rolling(3).mean() < 0)) | ((df['price_diff'].rolling(3).mean() < 0) &(df['rsi_diff'].rolling(3).mean() > 0))
#     df['divergence_flag'] = df['divergence_flag'].astype(int)
#     df['candle_body'] = abs(df['Close'] - df['Open']) / df['Close']
#     df['candle_dir'] = (df['Close'] > df['Open']).astype(int)
#     vol_mean = df['Volume'].rolling(10).mean()
#     vol_std  = df['Volume'].rolling(10).std()
#     df['volume_spike'] = (df['Volume'] > (vol_mean + vol_std)).astype(int)
#     df['vol_candle_signal'] = ((df['candle_dir'] == 1) &(df['volume_spike'] == 1)).astype(int)

#     df['Body_to_Range'] = (df['candle_body'] /(df['High'] - df['Low']).replace(0, np.nan))
#     df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
#     df['Rolling_Mean_Return'] = df['Log_Return'].rolling(5).mean()
#     df['Rolling_Std_Return'] = df['Log_Return'].rolling(5).std()
#     df['EMA_20_50_dist'] = (df['EMA20'] - df['EMA50']) / df['ATR']
#     df['Dist_from_EMA5'] = (df['Close'] - df['EMA5']) / df['ATR']
#     df['Trend_Strength'] = abs(df['Close'] - df['EMA5']) / df['ATR']
#     df['Dist_to_Recent_High'] = (df['High'].rolling(10).max() - df['Close']) / df['ATR']
#     df['Dist_to_Recent_Low'] = (df['Close'] - df['Low'].rolling(5).min()) / df['ATR']
#     df['Dist_to_Rolling_Max'] = (df['Close'].rolling(10).max() - df['Close']) / df['ATR']
#     df['Dist_to_Rolling_Min'] = (df['Close'] - df['Close'].rolling(10).min()) / df['ATR']
#     df["Rolling_Mean_Volume"] = df["Volume"].rolling(window=20).mean()
#     df["Volume_Spike"] = df["Volume"] / df["Rolling_Mean_Volume"]
#     df["Vol_Range"] = df["Volume"] * (df["High"] - df["Low"])
#     df['Volume_Z'] = ((df['Volume'] - df['Rolling_Mean_Volume']) /df['Volume'].rolling(10).std()).clip(-5, 5)

#     WINDOW = 12  # you can tune this
#     df['rolling_high'] = df['High'].rolling(WINDOW).max().shift(1)
#     df['rolling_low']  = df['Low'].rolling(WINDOW).min().shift(1)
#     df['dist_to_resistance'] = (df['rolling_high'] - df['Close']) / df['ATR']
#     df['dist_to_support'] = (df['Close'] - df['rolling_low']) / df['ATR']
#     df['near_resistance'] = (df['dist_to_resistance'] < 0.5).astype(int)
#     df['near_support'] = (df['dist_to_support'] < 0.5).astype(int)

#     LOOKBACK = 10
#     df['prev_high'] = df['High'].shift(1).rolling(LOOKBACK).max()
#     df['prev_low']  = df['Low'].shift(1).rolling(LOOKBACK).min()
#     df['broke_prev_high'] = (df['High'] > df['prev_high']).astype(int)
#     df['broke_prev_low'] = (df['Low'] < df['prev_low']).astype(int)
#     df['turtle_soup_sell'] = ((df['High'] > df['prev_high']) &(df['Close'] < df['prev_high'])).astype(int)
#     df['turtle_soup_buy'] = ((df['Low'] < df['prev_low']) &(df['Close'] > df['prev_low'])).astype(int)

#     STRUCT_WINDOW = 10
#     df['structure_high'] = df['High'].rolling(STRUCT_WINDOW).max().shift(1)
#     df['structure_low'] = df['Low'].rolling(STRUCT_WINDOW).min().shift(1)
#     df['bos_up'] = (df['Close'] > df['structure_high']).astype(int)
#     df['bos_down'] = (df['Close'] < df['structure_low']).astype(int)
#     df['structure_direction'] = np.where(df['bos_up'] == 1, 1,np.where(df['bos_down'] == 1, -1, 0))

#     FIB_WINDOW = 10
#     df['swing_high'] = df['High'].rolling(FIB_WINDOW).max().shift(1)
#     df['swing_low'] = df['Low'].rolling(FIB_WINDOW).min().shift(1)
#     fib_range = df['swing_high'] - df['swing_low']
#     df['fib_38'] = df['swing_high'] - 0.382 * fib_range
#     df['fib_50'] = df['swing_high'] - 0.5 * fib_range
#     df['fib_618'] = df['swing_high'] - 0.618 * fib_range
#     df['dist_fib_618'] = abs(df['Close'] - df['fib_618']) / df['ATR']
#     df['fib_618_zone'] = (df['dist_fib_618'] < 0.5).astype(int)
#     df['Range'] = df['High'] - df['Low'] + 1e-6
#     df['Body'] = abs(df['Close'] - df['Open'])
#     df['Candle_Strength'] = (df['Close'] - df['Open']) / df['Range']
#     df['Close_Position'] = (df['Close'] - df['Low']) / df['Range']
#     df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']
#     df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']
#     df['wick_ratio'] = df['Range'] / (df['Body'] + 1e-6)
#     df['PinBar_Bull'] = ((df['Lower_Wick'] > 0.6) &(df['Upper_Wick'] < 0.2)).astype(int)
#     df['PinBar_Bear'] = ((df['Upper_Wick'] > 0.6) &(df['Lower_Wick'] < 0.2)).astype(int)
#     df['Impulse_Bull'] = ((df['Body'] / df['Range'] > 0.6) &(df['Close'] > df['Open'])).astype(int)
#     df['Impulse_Bear'] = ((df['Body'] / df['Range'] > 0.6) &(df['Close'] < df['Open'])).astype(int)
#     df['Inside_Bar'] = ((df['High'] < df['High'].shift(1)) &(df['Low'] > df['Low'].shift(1))).astype(int)
#     df['Bull_Engulf'] = ((df['Close'] > df['Open']) &(df['Open'] < df['Close'].shift(1)) &(df['Close'] > df['Open'].shift(1))).astype(int)
#     df['Bear_Engulf'] = ((df['Close'] < df['Open']) &(df['Open'] > df['Close'].shift(1)) &(df['Close'] < df['Open'].shift(1))).astype(int)
#     df['Doji'] = (df['Body'] / df['Range'] < 0.1).astype(int)
#     df['Bull_Pressure'] = ((df['Close'] > df['Open']).rolling(3).mean())
#     df['Bear_Pressure'] = ((df['Close'] < df['Open']).rolling(3).mean())

#     PRICE_BUCKETS = 50
#     price_min = df['Low'].rolling(50).min()
#     price_max = df['High'].rolling(50).max()
#     vol_bucket = ((df['Close'] - price_min) / (price_max - price_min + 1e-6)) * (PRICE_BUCKETS - 1)
#     vol_bucket = vol_bucket.replace([np.inf, -np.inf], np.nan)
#     vol_bucket = vol_bucket.fillna(0)
#     vol_bucket = vol_bucket.clip(0, PRICE_BUCKETS - 1)
#     df['vol_bucket'] = vol_bucket.astype(int)

#     vol_node = df.groupby('vol_bucket')['Volume'].transform('sum')
#     df['volume_density'] = vol_node
#     df['dist_to_high_vol_node'] = df['High'].rolling(10).max() - df['Close']
    
#     df['tick_imbalance'] = (df['Close'] - df['Open']) * df['Volume']
#     df['rolling_buy_pressure'] = df['tick_imbalance'].clip(lower=0).rolling(10).sum()
#     df['rolling_sell_pressure'] = -df['tick_imbalance'].clip(upper=0).rolling(10).sum()
    
#     df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
#     df['rolling_vol'] = df['log_return'].rolling(10).std()
#     long_term_vol = df['log_return'].rolling(50).std()
#     df['volatility_state'] = (df['rolling_vol'] > long_term_vol).astype(int)
#     df['volatility_ratio'] = df['rolling_vol'] / (long_term_vol + 1e-6)

#     # EMA50 vs EMA200 on multiple timeframes (here we approximate with current df)
#     df['EMA50_5M'] = ta.trend.ema_indicator(df['Close'], window=50)
#     df['EMA200_5M'] = ta.trend.ema_indicator(df['Close'], window=200)
#     df['trend_5M'] = (df['EMA50_5M'] > df['EMA200_5M']).astype(int)
    
#     df['EMA50_15M'] = df['EMA50_5M'].rolling(3).mean()  # approximate higher TF by aggregation
#     df['EMA200_15M'] = df['EMA200_5M'].rolling(3).mean()
#     df['trend_15M'] = (df['EMA50_15M'] > df['EMA200_15M']).astype(int)
    
#     df['trend_alignment'] = (df['trend_5M'] == df['trend_15M']).astype(int)
#     df['candle_range'] = df['High'] - df['Low'] + 1e-6
#     df['wick_to_body_ratio'] = ((df['High'] - df[['Open','Close']].max(axis=1)) + 
#                                 (df[['Open','Close']].min(axis=1) - df['Low'])) / (abs(df['Close'] - df['Open']) + 1e-6)
#     df['candle_range_ratio'] = df['candle_range'] / (df['ATR'] + 1e-6)
#     df['open_close_ratio'] = abs(df['Open'] - df['Close']) / (df['candle_range'] + 1e-6)
#     df['gap_from_prev_close'] = (df['Open'] - df['Close'].shift(1)) / (df['ATR'].shift(1) + 1e-6)
    
#     lookback = 5
#     df['fractal_bull'] = ((df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(2))).astype(int)
#     df['fractal_bear'] = ((df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(2))).astype(int)
#     df['fractal_score'] = df['fractal_bull'] - df['fractal_bear']
#     df['fractal_score_smooth'] = df['fractal_score'].rolling(lookback).mean().fillna(0)
    
#     df['trend_slope'] = (df['Close'] - df['Close'].shift(lookback)) / lookback
#     df['vol_adj_trend'] = df['trend_slope'] / (df['ATR'] + 1e-6)  # normalized by ATR
    
#     window_entropy = 10
#     log_ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
#     def rolling_entropy(x):
#         hist, _ = np.histogram(x, bins=10, density=True)
#         return entropy(hist + 1e-8)  # small value to avoid log(0)
#     df['entropy'] = log_ret.rolling(window_entropy).apply(rolling_entropy, raw=True).fillna(0)
    
#     pca_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'ATR', 'RSI', 'EMA20', 'EMA50', 'EMA200']
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(df[pca_features].fillna(0))
#     pca = PCA(n_components=2)  # 2 latent features for simplicity
#     latent = pca.fit_transform(scaled_data)
#     df['latent_1'] = latent[:,0]
#     df['latent_2'] = latent[:,1]
#     df['vol_regime'] = (df['ATR'] > df['ATR'].rolling(50).mean()).astype(int)
    
#     ema_period = 50
#     slope_period = 5
#     df['EMA_50'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
#     df['EMA_slope'] = (df['EMA_50'] - df['EMA_50'].shift(slope_period)) / slope_period
    
#     volume_window = 20
#     volume_mean = df['Volume'].rolling(volume_window).mean()
#     volume_std = df['Volume'].rolling(volume_window).std()

#     df['volume_zscore'] = ((df['Volume'] - volume_mean) / (volume_std + 1e-9))
#     short_window = 10
#     long_window = 50
#     short_range = (df['High'].rolling(short_window).max() -df['Low'].rolling(short_window).min())
#     long_range = (df['High'].rolling(long_window).max() -df['Low'].rolling(long_window).min())

#     df['rolling_range_ratio'] = short_range / (long_range + 1e-9)

#     df['trend_regime'] = ((df['EMA50'] > df['EMA200']).astype(int) - (df['EMA50'] < df['EMA200']).astype(int))
#     df['dist_from_vwap'] = (df['Close'] - df['VWAP']) / df['ATR']
#     df['range_expansion'] = df['Range'] / df['Range'].rolling(20).mean()
#     hurst_window = 50

#     def hurst(ts):
#         lags = range(2, 20)
#         tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
#         poly = np.polyfit(np.log(lags), np.log(tau), 1)
#         return poly[0] * 2.0

#     df['hurst'] = df['Close'].rolling(hurst_window).apply(hurst, raw=True)
#     df['hurst10'] = df['Close'].rolling(10).apply(hurst, raw=True)
#     df['hurst20'] = df['Close'].rolling(20).apply(hurst, raw=True)
#     df['hurst30'] = df['Close'].rolling(30).apply(hurst, raw=True)
#     df['hurst40'] = df['Close'].rolling(40).apply(hurst, raw=True)
#     df['session_london'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 16)).astype(int)
#     df['session_newyork'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] <= 22)).astype(int)

#     df['order_flow_imbalance'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
#     df['return_1'] = df['Close'].pct_change()
#     df['volume_weighted_momentum'] = df['return_1'] * df['Volume']
#     df['vol_of_vol'] = df['ATR'].rolling(20).std()
#     df['momentum'] = df['Close'].pct_change(5)
#     df['price_acceleration'] = df['momentum'] - df['momentum'].shift(1)
#     mid_price = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
#     df['liquidity_vacuum'] = (df['Close'] - mid_price) / (df['ATR'] + 1e-9)

#     dir_window = 20
#     direction = abs(df['Close'] - df['Close'].shift(dir_window))
#     volatility = df['Close'].diff().abs().rolling(dir_window).sum()
#     df['trend_efficiency'] = direction / volatility
#     df['liquidity_sweep_high'] = ((df['High'] > df['rolling_high']) &
#         (df['Close'] < df['rolling_high'])).astype(int)
#     df['micro_pressure'] = ((df['Close'] - df['Low']) -(df['High'] - df['Close']))
#     df['micro_pressure_norm'] = df['micro_pressure'] / df['Range']
#     df['volatility_compression'] = df['ATR_6'] / df['ATR_36']

#     df['liquidity_sweep_low'] = (
#         (df['Low'] < df['rolling_low']) &
#         (df['Close'] > df['rolling_low'])).astype(int)
#     df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
#     df['realized_vol_20'] = (df['log_return'].rolling(20).apply(lambda x: np.sqrt(np.sum(x**2))))

#     window = 14
#     df['SMA'] = df['Close'].rolling(window).mean()
#     df['SMA_slope'] = df['SMA'].diff()
#     df['Trend_vs_Range'] = np.where(abs(df['SMA_slope']) > df['Close'].pct_change().rolling(window).std(), 1, 0)
#     # 1 = trend, 0 = range

#     df['Returns'] = df['Close'].pct_change()
#     df['Volatility'] = df['Returns'].rolling(window).std()

#     df['Hammer'] = np.where(
#         (df['High'] - df['Low'] > 3*(df['Open'] - df['Close'])) &
#         ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6), 1, 0)
#     df['Price_Change'] = df['Close'].diff()
#     df['Divergence'] = np.where((df['RSI'].diff() > 0) & (df['Price_Change'] < 0), 1, 0)
#     df['Overbought'] = np.where(df['RSI'] > 70, 1, 0)
#     df['Oversold'] = np.where(df['RSI'] < 30, 1, 0)

#     regime_features = [
#         'ATR',
#         'volatility_ratio',
#         'bollinger_width',
#         'trend_strength',
#         'ADX',
#         'EMA_slope',
#         'RSI',
#         'ROC',
#         'volume_spike',
#         'volume_zscore',
#         'close_position',
#         'rolling_range_ratio'
#     ]
#     df_regime = df[regime_features].dropna()

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df_regime)

#     # KMeans clustering
#     kmeans = KMeans(n_clusters=4, random_state=42)
#     regimes = kmeans.fit_predict(X_scaled)

#     df['market_regime'] = -1
#     df.loc[df_regime.index, 'market_regime'] = regimes

#     threshold = 0.7
#     nan_fraction = df.isna().mean()
#     cols_to_keep = nan_fraction[nan_fraction < threshold].index
#     df = df[cols_to_keep]
#     # print("Columns kept:", len(cols_to_keep))
#     # print("Columns dropped:", len(df.columns) - len(cols_to_keep))

#     df.set_index("Date", inplace=True)
#     return df




# # Initialize MT5 once
# if not mt5.initialize():
#     raise RuntimeError("❌ MT5 initialization failed")
# print("✅ MT5 initialized successfully")
 

# HIGH_TARGETS = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']

# BASE_PATH = "ALL_MODELS"

# def load_models(model_type, symbol):
#     models_dict = {}

#     for target in HIGH_TARGETS:
#         file_path = f"{BASE_PATH}/HL_{model_type}_{target}_{symbol}_model.pkl"

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Model not found: {file_path}")

#         bundle = joblib.load(file_path)

#         models_dict[target] = {
#             "model": bundle["model"],
#             "features": bundle["features"]
#         }

#         print(f"✅ Loaded {model_type} model for {target}")

#     return models_dict


# # LOAD ALL MODELS HERE
# MAIN_MODELS = load_models("MAIN", SYMBOL)
# HELPER_MODELS = load_models("HELPER", SYMBOL)
# # DIRECTION_MODELS = load_models("DIRECTION", SYMBOL)
# THRESH_MODELS = load_models("THRESH", SYMBOL)

# weights = {
# 'THL_5M'  : 1.5,
# 'THL_10M' : 1.4,
# 'THL_15M' : 1.1,
# 'THL_20M' : 1.0,
# 'THL_30M' : 0.7
# }

# model_weights = {
#     "main": 0.6,       # most reliable
#     # "dir": 0.2,
#     "helper": 0.4      # strict confirmation
# }

# W_threshold = 0.56 # DIRECTION 
# threshold = 0.02 # THRESHOLD (+/-)
# increaser = 0.5

# def EXTENT_TP():
#     # global mt5, pd, SYMBOL,apply_features, MAIN_MODELS, HELPER_MODELS, THRESH_MODELS,weights,model_weights,W_threshold,threshold

#     TIMEFRAME = mt5.TIMEFRAME_M5
#     N_BARS = 350

#     rates = mt5.copy_rates_from_pos(SYMBOL,TIMEFRAME,0,N_BARS )

#     if rates is None or len(rates) < N_BARS:
#         mt5.shutdown()
#         raise RuntimeError("❌ Failed to fetch enough closed candles")

#     data = pd.DataFrame(rates)
#     data['Date'] = pd.to_datetime(data['time'], unit='s')

#     data.rename(columns={
#         'open': 'Open',
#         'high': 'High',
#         'low': 'Low',
#         'close': 'Close',
#         'tick_volume': 'Volume'
#     }, inplace=True)

#     new_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
#     new_df.sort_values('Date', inplace=True)
#     new_df.reset_index(drop=True, inplace=True)

#     df = apply_features(new_df)
#     df.dropna(inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     up_moves = {}
#     down_moves = {}
#     thresh_predicts = []
#     for target in HIGH_TARGETS:

#         print(f'\n [[ CURRENTLY PREDICTING TARGET : {target} ]]')

#         main_model = MAIN_MODELS[target]["model"]
#         help_model = HELPER_MODELS[target]["model"]
#         thresh_model = THRESH_MODELS[target]["model"]

#         main_cols = MAIN_MODELS[target]["features"]
#         help_cols = HELPER_MODELS[target]["features"]
#         thresh_cols  = THRESH_MODELS[target]["features"]
        
#         current_candle = df.tail(1)

#         X_main = current_candle[main_cols]
#         X_help = current_candle[help_cols]
#         X_thresh = current_candle[thresh_cols]

#         # Safety check
#         assert set(main_cols) == set(X_main.columns)
#         assert set(help_cols) == set(X_help.columns)        
#         assert set(thresh_cols) == set(X_thresh.columns)

#         main_proba = main_model.predict_proba(X_main)[0]
#         # dir_proba  = dir_model.predict_proba(X_dir)[0]
#         help_proba = help_model.predict_proba(X_help)[0]
#         thres_proba = thresh_model.predict(X_thresh)[0]

#         thresh_predicts.append(thres_proba)

#         up_prob = (
#             main_proba[1] * model_weights["main"] +
#             help_proba[1] * model_weights["helper"]
#         )

#         down_prob = (
#             main_proba[0] * model_weights["main"] +
#             help_proba[0] * model_weights["helper"]
#         )

#         # Normalize (optional but safer)
#         total = up_prob + down_prob
#         up_prob /= total
#         down_prob /= total

#         up_moves[target] = round(up_prob * 100, 2)
#         down_moves[target] = round(down_prob * 100, 2)

#     probs_up = {k: v/100 for k, v in up_moves.items()}
#     probs_down = {k: v/100 for k, v in down_moves.items()}

#     row = df.iloc[-1]

#     total_weight = sum(weights.values())
#     weighted_up = sum(probs_up[k] * weights[k] for k in weights) / total_weight
#     weighted_down = sum(probs_down[k] * weights[k] for k in weights) / total_weight
#     thresh_moves_mean = round(sum(thresh_predicts) / len(thresh_predicts), 2)

#     momentum = abs(thresh_moves_mean)
#     direction = max(weighted_up, weighted_down)
#     model2_confidence = 1 if direction >= W_threshold + (W_threshold*increaser) else 0
#     return model2_confidence



# df = pd.DataFrame({'DIR':['BUY','SELL','SELL']})
# print((df['DIR'] == 'BUY').sum())



# # import os
# # import time
# # import joblib
# import pandas as pd
# from func import apply_features


# sl_mult = 1.5

# SYMBOL = 'EURUSD'
# data = pd.read_csv(f'CSV_FILES/MT5_10M_BT_{SYMBOL}_Exchange_Rate_Dataset.csv') 
# print('DATA FILE LENGTH BEFORE :', len(data))

# data['Date'] = pd.to_datetime(data['Date'])
# BAD_HOURS = [21, 22]

# data = data[~data['Date'].dt.hour.isin(BAD_HOURS)].reset_index(drop=True)
# # print('DATA FILE LENGTH AFTER FILTERING :', len(data))
# # print(dataz['Date'].dt.hour.value_counts())

# df = apply_features(data)
# df.dropna(inplace=True)
# print('\t DATASET LOADED SUCCESSFULLY ')
# # df.reset_index(drop=True, inplace=True)

# print("\n 10M OVERALL ATR")
# print(df["ATR"].describe())




# # Convert date column
# df["Date"] = pd.to_datetime(df["Date"])

# # Extract hour
# df["Hour"] = df["Date"].dt.hour

# # ===============================
# # Overall spread statistics
# # ===============================

# print("\nOVERALL SPREAD")
# print(df["Spread"].describe())

# # ===============================
# # Hourly statistics
# # ===============================

# hourly_stats = (
#     df.groupby("Hour")["Spread"]
#     .agg([
#         "count",
#         "mean",
#         "median",
#         "min",
#         "max",
#         "std"
#     ])
# )

# print("\nSPREAD BY HOUR")
# print(hourly_stats)

# # ===============================
# # 7PM - 10PM statistics
# # ===============================

# night_df = df[df["Hour"].between(19, 22)]

# print("\n7PM-10PM SPREAD")
# print(night_df["Spread"].describe())

# # ===============================
# # Normal trading hours
# # Example: 7AM - 4PM
# # ===============================

# day_df = df[df["Hour"].between(7, 16)]

# print("\n7AM-4PM SPREAD")
# print(day_df["Spread"].describe())






















# # Initialize MT5 once
# if not mt5.initialize():
#     raise RuntimeError("❌ MT5 initialization failed")
# print("✅ MT5 initialized successfully")
 

# # HIGH_TARGETS = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']
# HIGH_TARGETS = ['THL_5M','THL_10M','THL_15M']


# BASE_PATH = "ALL_MODELS"

# def load_models(model_type, symbol):
#     models_dict = {}

#     for target in HIGH_TARGETS:
#         file_path = f"{BASE_PATH}/HL_{model_type}_{target}_{symbol}_model.pkl"

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Model not found: {file_path}")

#         bundle = joblib.load(file_path)

#         models_dict[target] = {
#             "model": bundle["model"],
#             "features": bundle["features"]
#         }

#         print(f"✅ Loaded {model_type} model for {target}")

#     return models_dict


# def load_models_11(model_type, symbol):
#     models_dict = {}

#     for target in HIGH_TARGETS:
#         file_path = f"{BASE_PATH}/HL_11_{model_type}_{target}_{symbol}_model.pkl"

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Model not found: {file_path}")

#         bundle = joblib.load(file_path)

#         models_dict[target] = {
#             "model": bundle["model"],
#             "features": bundle["features"]
#         }

#         print(f"✅ Loaded {model_type} model for {target}")

#     return models_dict



# # LOAD ALL MODELS HERE
# HELPER_MODELS = load_models("HELPER", SYMBOL)
# _11_MODELS = load_models_11("HELPER", SYMBOL)

# weights = {
# 'THL_5M'  : 1.5,
# 'THL_10M' : 1.4,
# 'THL_15M' : 1.2
# }

# delay_time = 60*2
# risk_percent = 1

# W_threshold = 0.53 # DIRECTION

# PREV_PRED_PROB = ''

# try:
#     while True:
#         TIMEFRAME = mt5.TIMEFRAME_M5
#         N_BARS = 300
#         wait_for_new_candle(mt5,SYMBOL, TIMEFRAME)

#         rates = mt5.copy_rates_from_pos(SYMBOL,TIMEFRAME,1,N_BARS )

#         if rates is None or len(rates) < N_BARS:
#             mt5.shutdown()
#             raise RuntimeError("❌ Failed to fetch enough closed candles")

#         data = pd.DataFrame(rates)
#         data['Date'] = pd.to_datetime(data['time'], unit='s')
#         data.rename(columns={
#             'open': 'Open',
#             'high': 'High',
#             'low': 'Low',
#             'close': 'Close',
#             'tick_volume': 'Volume',
#             'spread': 'Spread'
#         }, inplace=True)

#         new_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']]
#         new_df.sort_values('Date', inplace=True)
#         new_df.reset_index(drop=True, inplace=True)
#         print(new_df.tail())

#         df = apply_features(new_df)
#         df.dropna(inplace=True)
#         df.reset_index(drop=True, inplace=True)

#         up_moves = {}
#         down_moves = {}

#         _11_up_moves = {}
#         _11_down_moves = {}
#         for target in HIGH_TARGETS:

#             print(f'\n [[ CURRENTLY PREDICTING TARGET : {target} ]]')

#             help_model = HELPER_MODELS[target]["model"]
#             _11_model = _11_MODELS[target]["model"]

#             help_cols = HELPER_MODELS[target]["features"]
#             _11_cols = _11_MODELS[target]["features"]
            
#             current_candle = df.tail(1)

#             X_help = current_candle[help_cols]
#             X_11 = current_candle[_11_cols]
#             # Safety check
#             assert set(help_cols) == set(X_help.columns)
#             assert set(_11_cols) == set(X_11.columns)

#             help_proba = help_model.predict_proba(X_help)[0]
#             _11_proba = _11_model.predict_proba(X_11)[0]

#             up_prob = help_proba[1]
#             down_prob = help_proba[0]

#             _11_up_prob = _11_proba[1]
#             _11_down_prob = _11_proba[0]

#             print('HELPER UP PROB : ',help_proba[1])
#             print('HELPER DOWN PROB : ',help_proba[0])

#             print('11 UP PROB : ',_11_proba[1])
#             print('11 DOWN PROB : ',_11_proba[0])


#             up_moves[target] = round(up_prob * 100, 2)
#             down_moves[target] = round(down_prob * 100, 2)

#             _11_up_moves[target] = round(_11_up_prob * 100, 2)
#             _11_down_moves[target] = round(_11_down_prob * 100, 2)


#             direction = "UP" if up_prob > down_prob else "DOWN"
#             print(f"HELP  : {help_proba}")
#             print(f"➡ FINAL → {direction} ({round(max(up_prob, down_prob)*100,2)}%)")

#         probs_up = {k: v/100 for k, v in up_moves.items()}
#         probs_down = {k: v/100 for k, v in down_moves.items()}

#         _11_probs_up = {k: v/100 for k, v in _11_up_moves.items()}
#         _11_probs_down = {k: v/100 for k, v in _11_down_moves.items()}


#         print('===================== ACCOUNT INFORMATIONS ==============================')
#         account_info = mt5.account_info()
#         balance = account_info.balance
#         print("Account Number:", account_info.login)
#         print("Balance:", account_info.balance)
#         print("Equity:", account_info.equity)
#         print("Free Margin:", account_info.margin_free)
#         print("Leverage:", account_info.leverage)  

#         pip_info = get_pip_info(mt5, SYMBOL)
#         pip_size = pip_info["pip_size"]
#         pip_value_per_lot = pip_info["pip_value_per_lot"]

#         # Get current tick
#         tick = mt5.symbol_info_tick(SYMBOL)
#         ask_price = tick.ask
#         bid_price = tick.bid
#         spread = ask_price - bid_price  # real-time spread

#         symbol_info = mt5.symbol_info(SYMBOL)
#         spread_points = spread / symbol_info.point
#         max_spread_points = 30   # example threshold

#         if spread_points > max_spread_points:
#             print(f"⚠️ Spread too high: {spread_points:.2f} points, skipping trade")
#             time.sleep(delay_time)
#             continue

#         print('pip_size:',pip_size)
#         print('pip_value_per_lot:',pip_value_per_lot)
#         print('ask_price:',ask_price)
#         print('bid_price:',bid_price)
#         print('spread:',spread)

#         row = df.iloc[-1]
#         # ATR_pips = row["ATR"] / pip_size

#         # # Apply ATR multiplier and enforce min/max limits
#         # SL_pips = ATR_pips * 1.0
#         # TP_pips = ATR_pips * 1.3

#         # # BUY trade SL/TP
#         # entry_buy = ask_price
#         # SL_buy = (entry_buy - (SL_pips * pip_size))
#         # TP_buy = (entry_buy + (TP_pips * pip_size))

#         # # SELL trade SL/TP
#         # entry_sell = bid_price
#         # SL_sell = (entry_sell + (SL_pips * pip_size))
#         # TP_sell = (entry_sell - (TP_pips * pip_size))

#         print('\n=======================================================================')

#         positions = mt5.positions_get(symbol=SYMBOL)

#         if positions is None:
#             print("⚠️ Error fetching positions")
#             has_open_trade = False
#             has_buy = False
#             has_sell = False

#         elif len(positions) == 0:
#             has_open_trade = False
#             has_buy = False
#             has_sell = False

#         else:
#             buy_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
#             sell_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

#             has_buy = len(buy_positions) > 0
#             has_sell = len(sell_positions) > 0
#             has_open_trade = has_buy or has_sell   # ✅ FIX

#             if has_buy:
#                 print("⚠️ BUY exists")

#             if has_sell:
#                 print("⚠️ SELL exists")

                
#         total_weight = sum(weights.values())
#         weighted_up = sum(probs_up[k] * weights[k] for k in weights) / total_weight
#         weighted_down = sum(probs_down[k] * weights[k] for k in weights) / total_weight

#         _11_weighted_up = sum(_11_probs_up[k] * weights[k] for k in weights) / total_weight
#         _11_weighted_down = sum(_11_probs_down[k] * weights[k] for k in weights) / total_weight


#         if PREV_PRED_PROB == weighted_up or PREV_PRED_PROB == weighted_down:
#             print('\n CURRENT CANDLE PREDICTION SAME AS BEFORE')
#             time.sleep(delay_time)
#             continue

#         print('\n================ WEIGHTED RESULT =================')
#         print(f'WEIGHTED UP   : {round(weighted_up*100,2)}%')
#         print(f'WEIGHTED DOWN : {round(weighted_down*100,2)}%')
#         print(f'_11_WEIGHTED UP   : {round(_11_weighted_up*100,2)}%')
#         print(f'_11_WEIGHTED DOWN : {round(_11_weighted_down*100,2)}%')

#         if not has_open_trade:
#             ##================================= MAIN DIRECTION TRADING ==================================================
#             if weighted_up >= W_threshold-2 and _11_weighted_up >= W_threshold :
#                 print('CHECKING FOR BUY ENTRY LOGIC')
#                 entry_buy = Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
#                     direction="BUY",atr_value=row["ATR"])

#                 if entry_buy is None:
#                     print("❌ BUY skipped (no confirmation)")
#                 else:
#                     PREV_PRED_PROB = weighted_up
#                     print("🚀 FINAL SIGNAL: BUY")

#                     atr = row["ATR"]
#                     SL_distance = atr * 1.0
#                     TP_distance = atr * 1.3
#                     SL_pips = SL_distance/ pip_size 

#                     SL_buy = entry_buy - SL_distance
#                     TP_buy = entry_buy + TP_distance
#                     lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
#                     print(f'LOTS SIZE USED: {lot_size}')

#                     result = place_buy(mt5, SYMBOL, lot_size, entry_buy, SL_buy, TP_buy)
#                     print("BUY ORDER RESULT:", result)
#                     log_trade(mt5=mt5,symbol=SYMBOL,direction="BUY",entry_price=entry_buy,SL=SL_buy,TP=TP_buy,lot_size=lot_size,
#                         proba_up=weighted_up,proba_down=weighted_down,order_result=result)

#             elif weighted_down >= W_threshold  and _11_weighted_down >= W_threshold :
#                 print('CHECKING FOR SELL ENTRY LOGIC')
#                 entry_sell = Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
#                     direction="SELL",atr_value=row["ATR"])

#                 if entry_sell is None:
#                     print("❌ SELL skipped (no confirmation)")
#                 else:
#                     PREV_PRED_PROB = weighted_down
#                     print("🔻 FINAL SIGNAL: SELL")

#                     atr = row["ATR"]
#                     SL_distance = atr * 1.3
#                     TP_distance = atr * 1.3
#                     SL_pips = SL_distance/ pip_size 

#                     SL_sell = entry_sell + SL_distance
#                     TP_sell = entry_sell - TP_distance
#                     lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
#                     print(f'LOTS SIZE USED: {lot_size}')

#                     result = place_sell(mt5, SYMBOL, lot_size, entry_sell, SL_sell, TP_sell)
#                     print("SELL ORDER RESULT:", result)
#                     log_trade(mt5=mt5,symbol=SYMBOL,direction="SELL",entry_price=entry_sell,
#                         SL=SL_sell,TP=TP_sell,lot_size=lot_size,proba_up=weighted_up,proba_down=weighted_down,order_result=result)
                
#             else:
#                 print("⏳ FINAL SIGNAL: NO TRADE")

#             while True:
#                 print('CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION')
#                 atr = row["ATR"]
#                 output = move_sl_and_partial_close(mt5 = mt5, SYMBOL = SYMBOL, atr_value = atr)
#                 if output == 1:
#                     print('TP/SL + PARTIAL CONDITION MET')
#                     break
#                 time.sleep(1)

                
#         elif has_open_trade:
#             while True:
#                 print('CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION')
#                 atr = row["ATR"]
#                 output = move_sl_and_partial_close(mt5 = mt5, SYMBOL = SYMBOL, atr_value = atr)
#                 if output == 1:
#                     print('TP/SL + PARTIAL CONDITION MET')
#                     break
#                 time.sleep(1)


# finally:
#     # Shutdown MT5 only once, when the bot stops
#     mt5.shutdown()
#     print("✅ MT5 shutdown successfully")





# print(6 > 3+3)

# EURUSD
# DATA FILE LENGHT : 653841
# after APPLY FEATURES : 653841
# after apply dropna: 653841
# COLUMNS: 7
# ROWS: 653841
# Label Distribution:
# THL_1H THL_1H
#  0    0.406280
#  1    0.397092
# -1    0.196628
# Name: proportion, dty


# USDJPY
# DATA FILE LENGHT : 545708
# after APPLY FEATURES : 545708
# after apply dropna: 545708
# COLUMNS: 7
# ROWS: 545708
# Label Distribution:
# THL_1H THL_1H
#  1    0.403297
#  0    0.402696
# -1    0.194008
# Name: proportion, dtype: float64


# GBPUSD
# DATA FILE LENGHT : 471370
# after APPLY FEATURES : 471370
# after apply dropna: 471370
# COLUMNS: 7
# ROWS: 471370
# Label Distribution:
# THL_1H THL_1H
#  0    0.406405
#  1    0.399669
# -1    0.193925
# Name: proportion, dtype: float64


# XAUUSD
# DATA FILE LENGHT : 510495
# after APPLY FEATURES : 510495
# after apply dropna: 510495
# COLUMNS: 7
# ROWS: 510495
# Label Distribution:
# THL_1H THL_1H
#  1    0.397865
#  0    0.396333
# -1    0.205802
# Name: proportion, dtype: float64

# XAUUSD
# After apply dropna: 510289
# COLUMNS: 222
# ROWS: 510289
# Label Distribution:
# THL_130M THL_130M
#  1    0.406859
#  0    0.402228
# -1    0.190913
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_130M
# ==============================
# [I 2026-05-10 16:14:31,517] A new study created in memory with name: no-name-a287f381-966a-487a-a361-a78e01609811
# THL_130M
# 1    0.502862
# 0    0.497138
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(1.0057565348729567), np.int64(1): np.float64(0.9943089861662364)}
# LEN X AFTER PREPROCESSING :  412853
# THL_130M
# 1    0.502862
# 0    0.497138
# Name: proportion, 



# from multiprocessing import Process
# from TRADE_HIGH_NXT import FOREX_TRADING


# def run_symbol(symbol):
#     print(f'CURRENTLY RUNNING THIS SYMBOL [[ {symbol} ]]')
#     FOREX_TRADING(SYMBOL=symbol)


# if __name__ == "__main__":

#     symbols = ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD"]

#     processes = []

#     for symbol in symbols:
#         p = Process(target=run_symbol, args=(symbol,))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()




# import time
# from datetime import datetime, timedelta

# now = datetime.now()

# minute = now.minute
# second = now.second

# print('now',now)
# print('minute',minute)
# print('second',second)


# # Remaining minutes to next 5-minute mark
# remaining_minutes = 5 - (minute % 5)
# print('remaining_minutes',remaining_minutes)

# # Fix exact multiple of 5 issue
# if remaining_minutes == 5:
#     remaining_minutes = 0

# # Seconds remaining to next candle
# seconds_remaining = (remaining_minutes * 60) - second
# print('seconds_remaining',seconds_remaining)

# # If less than 15 seconds remain,
# # begin precise MT5 checking
# # if seconds_remaining <= 15:
# #     print("\n[NEAR NEXT CANDLE -> STARTING MT5 CHECK]\n")
# #     break

# # Sleep efficiently
# sleep_time = max(seconds_remaining - 10, 1)

# print(
#     f"Efficient waiting... "
#     f"Next candle in {seconds_remaining}s | "
#     f"Sleeping {sleep_time}s"
# )

# time.sleep(sleep_time)


# SYMBOL = 'AUDCAD'
# Early stopping, best iteration is:
# [11]	valid_0's auc: 0.568059	valid_0's binary_logloss: 0.683007
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5681




# SYMBOL = 'USDCAD'
# Early stopping, best iteration is:
# [4]	valid_0's auc: 0.530313	valid_0's binary_logloss: 0.690612
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5303



# SYMBOL = 'NZDUSD'
# Early stopping, best iteration is:
# [59]	valid_0's auc: 0.552268	valid_0's binary_logloss: 0.682182
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5523




#USDCNH, USDSEK,USDCHF
# "EURUSD", "USDJPY", "GBPUSD", "XAUUSD", 'USDCAD', 'AUDCAD', 'NZDUSD', 'AUDUSD', 'USDCHF'

# AFTER BACKTESTING WITHOUT CONSIDERING:
# 1) NO PARTIAL PROFIT
# 2) NO MOVING SL TO BREAK EVEN 
# 3) NO FILTERING ENTRYING
# 4) NO ANYTHING LIKE RISK MANAGEMENT
# 5) LOSS IS WAS BEEN CONSIDERED FIRST INCASE OF BOTH TP AND SL IS HIT, I ASSUMED LOSS
# JUST PURE TP AND SL HIT,SPREAD WAS ALSO CONSIDERED
# TO MY SUPRISE, ALL THE FOREX CURRENCY PAIRS PERFORMED REALLY WELL WITH A  GIVEN THRESHOLD FOR THE CURRENCY AND EXCLUDING GOLD(XAUUSD).
# USING 1-3 MONTH DATA FOR BACTESTING FOR ALL CURRENCY PAIRS ON CAPIATL =  $1,000

# 1) EURUSD 
# THRESDOLD = 54%
# PROFIT = $189-220/MONTH 
# NUMBER OF TRADES = 70

# 2) USDJPY 
# THRESDOLD = 54%
# PROFIT = $150-190/MONTH 
# NUMBER OF TRADES = 68

# 3) GBPUSD 
# THRESDOLD = 54%
# PROFIT = $150-200/MONTH 
# NUMBER OF TRADES = 75

# 4) USDCAD 
# THRESDOLD = 54%
# PROFIT = $150-230/MONTH 
# NUMBER OF TRADES = 65

# 5) AUDCAD 
# THRESDOLD = 54%
# PROFIT = $143-180/MONTH 
# NUMBER OF TRADES = 71

# 6) NZDUSD 
# THRESDOLD = 57%
# PROFIT = $140-170/MONTH 
# NUMBER OF TRADES = 89

# 7) AUDUSD 
# THRESDOLD = 54%
# PROFIT = $185-230/MONTH 
# NUMBER OF TRADES = 57




# from google.colab import files
# uploaded = files.upload()

# pip install pandas numpy ta lightgbm shap joblib optuna

# import sys
# import os

# sys.path.append(os.path.abspath('/content/FOREX TRADING'))


# balance = 1000
# risk_percent = 1
# pip_value_per_lot = 10.0
# vol_info = {'min': 0.01, 'max': 500.0, 'step': 0.01}

# df = pd.DataFrame({'a':[12,4,5,8,9,34,5,6]})
# # print(10/df['a'])

# df['LOTS'] = calc_lot_size(balance,risk_percent,df['a'],pip_value_per_lot=pip_value_per_lot,min_lot=0.01,max_lot=2)


# CURRENTLY RUNNING SYMBOL IS AUDUSD
# DATA FILE LENGHT : 315700
# after APPLY FEATURES : 315700
# after apply dropna: 315494
# COLUMNS: 222
# ROWS: 315494
# Label Distribution:
# THL_2H THL_2H
#  0    0.480532
#  1    0.465773
# -1    0.053695
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_2H
# ==============================
# [I 2026-05-22 04:39:13,640] A new study created in memory with name: no-name-65f86572-94bf-4956-8acc-77a59f8022fb
# THL_2H
# 0    0.507798
# 1    0.492202
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.9846430903801626), np.int64(1): np.float64(1.0158435247759243)}
# LEN X AFTER PREPROCESSING :  298530
# THL_2H
# 0    0.507798
# 1    0.492202

# Early stopping, best iteration is:
# [6]	valid_0's auc: 0.54579	valid_0's binary_logloss: 0.690464
# 💾 FULL MODEL SAVED: THL_2H | AUC: 0.5458


# CURRENTLY RUNNING SYMBOL IS AUDCAD
# DATA FILE LENGHT : 458975
# after APPLY FEATURES : 458975
# after apply dropna: 458769
# COLUMNS: 222
# ROWS: 458769
# Label Distribution:
# THL_2H THL_2H
#  0    0.472279
#  1    0.460584
# -1    0.067138
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_2H
# ==============================
# [I 2026-05-22 04:41:14,045] A new study created in memory with name: no-name-90ff7c87-d98d-43b9-a665-bdd2d7c3c3c2
# THL_2H
# 0    0.506268
# 1    0.493732
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.9876185640765272), np.int64(1): np.float64(1.0126958209096502)}
# LEN X AFTER PREPROCESSING :  427945
# THL_2H
# 0    0.506268
# 1    0.493732

# Early stopping, best iteration is:
# [116]	valid_0's auc: 0.570629	valid_0's binary_logloss: 0.679462
# 💾 FULL MODEL SAVED: THL_2H | AUC: 0.5706


# CURRENTLY RUNNING SYMBOL IS CADJPY
# DATA FILE LENGHT : 668969
# after APPLY FEATURES : 668969
# after apply dropna: 668763
# COLUMNS: 213
# ROWS: 668763
# Label Distribution:
# THL_1H THL_1H
#  1    0.405744
#  0    0.403326
# -1    0.190931
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-25 21:30:02,311] A new study created in memory with name: no-name-c8242385-4207-4b06-b7aa-70cd6af08faa
# THL_1H
# 1    0.501494
# 0    0.498506
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(1.0029975085643101), np.int64(1): np.float64(0.997020354461729)}
# LEN X AFTER PREPROCESSING :  541065
# THL_1H
# 1    0.501494
# 0    0.498506

# Early stopping, best iteration is:
# [19]	valid_0's auc: 0.540608	valid_0's binary_logloss: 0.6835
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5406

# CURRENTLY RUNNING SYMBOL IS AUDJPY
# DATA FILE LENGHT : 668262
# after APPLY FEATURES : 668262
# after apply dropna: 668056
# COLUMNS: 213
# ROWS: 668056
# Label Distribution:
# THL_1H THL_1H
#  1    0.402720
#  0    0.400970
# -1    0.196311
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-25 20:36:17,396] A new study created in memory with name: no-name-b75a12b4-a431-4bac-a80d-095848a88523
# THL_1H
# 1    0.501089
# 0    0.498911
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(1.0021820693259664), np.int64(1): np.float64(0.9978274121486503)}
# LEN X AFTER PREPROCESSING :  536899
# THL_1H
# 1    0.501089
# 0    0.498911

# [24]	valid_0's auc: 0.545291
# 🏆 BEST AUC: 0.5429997504777487
# 🏆 BEST PARAMS: {'n_estimators': 595, 'learning_rate': 0.03493698800696091, 'num_leaves': 20, 'max_depth': 8, 'min_child_samples': 94, 'subsample': 0.9592849507821758, 'colsample_bytree': 0.8664507674360722, 'reg_alpha': 0.006665123172561454, 'reg_lambda': 0.296952656732291}

# Early stopping, best iteration is:
# [30]	valid_0's auc: 0.560015	valid_0's binary_logloss: 0.681793
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5600



# import ta
# import shap
# import joblib
# import optuna
# import numpy as np
# import pandas as pd
# from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# from sklearn.model_selection import TimeSeriesSplit
# from func import apply_features
# from sklearn.utils.class_weight import compute_class_weight

# CURRENTLY RUNNING SYMBOL IS EURJPY
# DATA FILE LENGHT : 668846
# after APPLY FEATURES : 668846
# after apply dropna: 668640
# COLUMNS: 213
# ROWS: 668640
# Label Distribution:
# THL_1H THL_1H
#  1    0.399899
#  0    0.395376
# -1    0.204726
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-25 21:33:12,378] A new study created in memory with name: no-name-52b79f5c-4e04-4660-a7f2-6566ee58dca2
# THL_1H
# 1    0.502843
# 0    0.497157
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(1.0057194950805532), np.int64(1): np.float64(0.9943451902327373)}
# LEN X AFTER PREPROCESSING :  531742
# THL_1H
# 1    0.502843
# 0    0.497157

# 🏆 BEST AUC: 0.5407868971935814
# 🏆 BEST PARAMS: {'n_estimators': 480, 'learning_rate': 0.03857826938712784, 'num_leaves': 113, 'max_depth': 4, 'min_child_samples': 73, 'subsample': 0.8699767518700213, 'colsample_bytree': 0.9851977002606847, 'reg_alpha': 0.017099576698074263, 'reg_lambda': 0.022229996122492872}

# Early stopping, best iteration is:
# [28]	valid_0's auc: 0.542612	valid_0's binary_logloss: 0.686908
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5426






# CURRENTLY RUNNING SYMBOL IS EURGBP
# DATA FILE LENGHT : 667667
# after APPLY FEATURES : 667667
# after apply dropna: 667461
# COLUMNS: 213
# ROWS: 667461
# Label Distribution:
# THL_1H THL_1H
#  0    0.395963
#  1    0.385542
# -1    0.218495
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-25 21:31:54,635] A new study created in memory with name: no-name-ed13404e-bbeb-4d43-99e2-f6c7377187ec
# THL_1H
# 0    0.506668
# 1    0.493332
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.9868399644323363), np.int64(1): np.float64(1.0135157716386416)}
# LEN X AFTER PREPROCESSING :  521614
# THL_1H
# 0    0.506668
# 1    0.493332
# 🏆 BEST AUC: 0.5684743314053347
# 🏆 BEST PARAMS: {'n_estimators': 495, 'learning_rate': 0.034530021330777365, 'num_leaves': 104, 'max_depth': 3, 'min_child_samples': 64, 'subsample': 0.6442108765612454, 'colsample_bytree': 0.6568603923969759, 'reg_alpha': 8.966384238657128, 'reg_lambda': 0.015625083907624578}
# [
# Early stopping, best iteration is:
# [219]	valid_0's auc: 0.583694	valid_0's binary_logloss: 0.676534
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5837


# CURRENTLY RUNNING SYMBOL IS GBPJPY
# DATA FILE LENGHT : 741791
# after APPLY FEATURES : 741791
# after apply dropna: 741585
# COLUMNS: 213
# ROWS: 741585
# Label Distribution:
# THL_1H THL_1H
#  1    0.399383
#  0    0.397276
# -1    0.203341
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-25 21:29:47,955] A new study created in memory with name: no-name-5a497e90-b958-43ea-a096-0851ff8b6347
# THL_1H
# 1    0.501322
# 0    0.498678
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(1.0026509712873672), np.int64(1): np.float64(0.9973630098828042)}
# LEN X AFTER PREPROCESSING :  590780
# THL_1H
# 1    0.501322
# 0    0.498678

# Early stopping, best iteration is:
# [55]	valid_0's auc: 0.536751
# 🏆 BEST AUC: 0.5427586448260169
# 🏆 BEST PARAMS: {'n_estimators': 659, 'learning_rate': 0.019709221876956524, 'num_leaves': 74, 'max_depth': 6, 'min_child_samples': 11, 'subsample': 0.7840117948335065, 'colsample_bytree': 0.6408768901923412, 'reg_alpha': 0.005315509479333024, 'reg_lambda': 7.364729946086882}
# [

# Early stopping, best iteration is:
# [28]	valid_0's auc: 0.54544	valid_0's binary_logloss: 0.689662
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5454



# CURRENTLY RUNNING SYMBOL IS AUDUSD
# DATA FILE LENGHT : 744242
# after APPLY FEATURES : 744242
# after apply dropna: 744036
# COLUMNS: 213
# ROWS: 744036
# Label Distribution:
# THL_1H THL_1H
#  0    0.411116
#  1    0.407125
# -1    0.181759
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-25 20:46:21,736] A new study created in memory with name: no-name-62b99aa1-dafe-49cf-8237-48cb30a5d874
# THL_1H
# 0    0.502439
# 1    0.497561
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.9951451549627305), np.int64(1): np.float64(1.0049024462711698)}
# LEN X AFTER PREPROCESSING :  608790
# THL_1H
# 0    0.502439
# 1    0.497561

# 🏆 BEST AUC: 0.5378224401794212
# 🏆 BEST PARAMS: {'n_estimators': 399, 'learning_rate': 0.01949739406960142, 'num_leaves': 152, 'max_depth': 4, 'min_child_samples': 20, 'subsample': 0.9144675570272546, 'colsample_bytree': 0.8500858948646517, 'reg_alpha': 0.10776959155283826, 'reg_lambda': 0.1816096992195228}

# Early stopping, best iteration is:
# [139]	valid_0's auc: 0.552439	valid_0's binary_logloss: 0.679941
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5524


# CURRENTLY RUNNING SYMBOL IS USDCHF
# DATA FILE LENGHT : 666672
# after APPLY FEATURES : 666672
# after apply dropna: 666466
# COLUMNS: 213
# ROWS: 666466
# Label Distribution:
# THL_1H THL_1H
#  0    0.408222
#  1    0.405732
# -1    0.186046
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-26 04:46:41,050] A new study created in memory with name: no-name-3fda3843-4cd7-410a-b558-7b8c86a1b5f2
# THL_1H
# 0    0.50153
# 1    0.49847
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(0.9969492135954804), np.int64(1): np.float64(1.0030695152754612)}
# LEN X AFTER PREPROCESSING :  542462
# THL_1H
# 0    0.50153
# 1    0.49847
# 🏆 BEST AUC: 0.5521918735336864
# 🏆 BEST PARAMS: {'n_estimators': 509, 'learning_rate': 0.023068371298932123, 'num_leaves': 229, 'max_depth': 5, 'min_child_samples': 56, 'subsample': 0.8882992373342512, 'colsample_bytree': 0.8495275280127099, 'reg_alpha': 0.5812211757446318, 'reg_lambda': 9.716337725043827}

#  Early stopping, best iteration is:
# [55]	valid_0's auc: 0.538529	valid_0's binary_logloss: 0.68445
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5385



# CURRENTLY RUNNING SYMBOL IS USDCAD
# DATA FILE LENGHT : 817988
# after APPLY FEATURES : 817988
# after apply dropna: 817782
# COLUMNS: 213
# ROWS: 817782
# Label Distribution:
# THL_1H THL_1H
#  1    0.409448
#  0    0.408625
# -1    0.181927
# Name: proportion, dtype: float64

# ==============================
# 🚀 FULL FEATURE TRAINING: THL_1H
# ==============================
# [I 2026-05-26 04:51:10,901] A new study created in memory with name: no-name-c17a620c-ccfb-4191-a820-f9346f161cd2
# THL_1H
# 1    0.500503
# 0    0.499497
# Name: proportion, dtype: float64
# Class Weights: {np.int64(0): np.float64(1.0010069996199438), np.int64(1): np.float64(0.9989950244001505)}
# LEN X AFTER PREPROCESSING :  668995
# THL_1H
# 1    0.500503
# 0    0.499497

# [I 2026-05-26 06:41:40,779] Trial 29 finished with value: 0.5360969547775888 and parameters: {'n_estimators': 362, 'learning_rate': 0.033062327240404726, 'num_leaves': 277, 'max_depth': 4, 'min_child_samples': 71, 'subsample': 0.9049688408704442, 'colsample_bytree': 0.8379925330562815, 'reg_alpha': 4.341222352523372, 'reg_lambda': 0.006179367513313877}. Best is trial 27 with value: 0.5365945578212094.
# 🏆 BEST AUC: 0.5365945578212094
# 🏆 BEST PARAMS: {'n_estimators': 334, 'learning_rate': 0.03004046005837962, 'num_leaves': 284, 'max_depth': 3, 'min_child_samples': 50, 'subsample': 0.966821579980583, 'colsample_bytree': 0.7606853810115631, 'reg_alpha': 4.303805512412305, 'reg_lambda': 0.006182689492760486}
# [Lig

# Early stopping, best iteration is:
# [134]	valid_0's auc: 0.535563	valid_0's binary_logloss: 0.687364
# 💾 FULL MODEL SAVED: THL_1H | AUC: 0.5356









# MODELS TO WORK ON

# AUDJPY 
# AUDUSD - UPGRADE 
# CADJPY 
# EURGBP 
# EURJPY 

# GBPJPY 

# USDCAD - UPGRADE
# USDCHF - UPGRADE 

# AUDJPY, AUDUSD, CADJPY, EURGBP, EURJPY, GBPJPY, USDCAD, USDCHF


# NOT TICKVOL = AUDCAD, EURUSD, GBPUSD, NZDUSD , USDJPY

# a = ['AUDCAD', 'EURUSD', 'GBPUSD', 'NZDUSD' , 'USDJPY']

# symbol = 'AUDCA1D'
# if symbol in a:
#     print('Yes')

# SYMBOL = 'EURUSDm'
# if 'EURUSD' in SYMBOL  :
#     print('yes')
# else:
#     print('No')

# import re
# def normalize_symbol(symbol):
#     return re.match(r"[A-Z]+",symbol).group()

# print(normalize_symbol('EURUSDmY'))



# =====================================================
# HOURLY STATS
# =====================================================

# df['Hour'] = pd.to_datetime(df['DATE']).dt.hour

# hour_stats = (
#     df.groupby('Hour')
#       .agg(
#           Trades=('RESULT','count'),
#           Wins=('RESULT', lambda x: (x=='WIN').sum())
#       )
# )

# hour_stats['WinRate'] = (
#     hour_stats['Wins']
#     /
#     hour_stats['Trades']
# ) * 100



# signals = working_df[
#     (working_df["UP_AVG"] >= 0.53) |
#     (working_df["DN_AVG"] >= 0.53)
# ].copy()

# signals["CONFIDENCE"] = np.maximum(
#     signals["UP_AVG"],
#     signals["DN_AVG"]
# )


# signals["Hour"] = signals.index.hour

# print(
#     signals.groupby("Hour")["CONFIDENCE"]
#     .agg(["count","mean","max"])
#     .sort_values("mean", ascending=False)
# )

# symbol = ["AUDCAD", "AUDUSD","AUDJPY", "CADJPY","EURUSD",'EURJPY','GBPJPY',"NZDUSD", "USDJPY","USDCAD"]
# symbol = [x+'m' for x in symbol]
# print(symbol)

# import pandas as pd





# df = pd.read_csv('CSV_FILES/MT5_5M_EURUSD_Exchange_Rate_Dataset.csv')

# print(df["Spread"].describe())
# print(df["Spread"].value_counts().head(20))



# def run_backtest(working_df, threshold):

#     test_df = working_df.copy()

#     test_df['SIGNAL'] = 0

#     test_df.loc[test_df['UP_AVG'] >= threshold,'SIGNAL'] = 1

#     test_df.loc[test_df['DN_AVG'] >= threshold,'SIGNAL'] = -1

#     wins = 0
#     losses = 0
#     trades = 0

#     i = 0

#     while i < len(test_df) - 1:

#         row = test_df.iloc[i]

#         if row['SIGNAL'] == 0:
#             i += 1
#             continue

#         trades += 1

#         trade_taken = False

#         for j in range(i + 1, len(test_df)):

#             high = test_df.iloc[j]['High']
#             low = test_df.iloc[j]['Low']

#             if row['SIGNAL'] == 1:

#                 if low <= row['SL_BUY']:
#                     losses += 1
#                     i = j
#                     trade_taken = True
#                     break

#                 elif high >= row['TP_BUY']:
#                     wins += 1
#                     i = j
#                     trade_taken = True
#                     break

#             elif row['SIGNAL'] == -1:

#                 if high >= row['SL_SELL']:
#                     losses += 1
#                     i = j
#                     trade_taken = True
#                     break

#                 elif low <= row['TP_SELL']:
#                     wins += 1
#                     i = j
#                     trade_taken = True
#                     break

#         if not trade_taken:
#             i += 1

#     total = wins + losses

#     win_rate = wins / total if total else 0

#     return {
#         "THRESHOLD": round(threshold, 3),
#         "TRADES": total,
#         "WINS": wins,
#         "LOSSES": losses,
#         "WIN_RATE": round(win_rate, 4)
#     }



# ========================== ATR * 1.5 FOR 1H-2H =======================================
# 'AUDCAD' =1H-59-0.585,AUDJPY = 1H-56-0.63,'AUDUSD' = 2H-76-0.528, BTCUSD = NONE,CADJPY = 2H-64-0.518, ETHUSD = NONE,EURJPY = NONE, 
#  "EURUSD" = 1H-57-0.61,GBPJPY = NONE,
#  "GBPUSD"=1H-61-0.63, NZDCAD = 1H-68-0.65, 'NZDUSD' = 1H-67-0.655,'USDCAD'=1H-58-0.62, 'USDCHF' = NONE, "USDJPY"=1H-57-0.56
#  "USDSEK" =1H-101-59 , 'XAGUSD'=NONE, "XAUUSD"=2H-0.52





# AUDCAD =0.68,70 ,AUDUSD,EURJPY

# SYMBOL: AUDCAD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.608      50    35      15    0.7000

# SYMBOL: BTCUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.514     178    95      83    0.5337

# SYMBOL: AUDJPY
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.593      54    36      18    0.6667

# SYMBOL: AUDUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.530      70    48      22    0.6857

# SYMBOL: CADJPY
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.641      50    37      13    0.7400

# SYMBOL: ETHUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.508     166    91      75    0.5482

# SYMBOL: EURJPY
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.716      64    42      22    0.6562

# SYMBOL: EURUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.521      66    48      18    0.7273

# SYMBOL: GBPJPY
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.617      58    38      20    0.6552

# SYMBOL: GBPUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.560      57    37      20    0.6491

# SYMBOL: NZDCAD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.765      53    36      17    0.6792

# SYMBOL: NZDUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.638      57    42      15    0.7368

# SYMBOL: USDCAD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.501     364   174     190    0.4780

# SYMBOL: USDCHF
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.463     425   215     210    0.5059

# SYMBOL: USDJPY
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.565      91    60      31    0.6593

# SYMBOL: USDSEK
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.702      79    52      27    0.6582

# SYMBOL: XAGUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.502     103    56      47    0.5437

# SYMBOL: XAUUSD
#  THRESHOLD  TRADES  WINS  LOSSES  WIN_RATE
#      0.502     160    95      65    0.5938



# ============================================================
# SYMBOL             : EURUSD
# LENGHT OF DATA SEARCHED IS : 6006
# Final Balance      : 253179.60
# MODEL TARGET       : ['THL_3H']
# Threshold Used     : 0.521
# Total Trades       : 66
# tp_mult * ATR      : 2.25
# sl_mult * ATR      : 2.25
# BUY                 :  14
# SELL                :  52
# Total Wins         : 48
# Total Losses       : 18
# Win Rate           : 72.73%
# Profit Factor      : 2.54
# Expectancy         : 805.75
# Max Drawdown       : 3.71%
# Sharpe Ratio       : 7.58
# ============================================================

# ============================================================
# SYMBOL             : AUDCAD
# LENGHT OF DATA SEARCHED IS : 5947
# Final Balance      : 236008.01
# MODEL TARGET       : ['THL_3H']
# Threshold Used     : 0.608
# Total Trades       : 50
# tp_mult * ATR      : 2.25
# sl_mult * ATR      : 2.25
# BUY                 :  4
# SELL                :  46
# Total Wins         : 35
# Total Losses       : 15
# Win Rate           : 70.00%
# Profit Factor      : 2.20
# Expectancy         : 720.16
# Max Drawdown       : 1.86%
# Sharpe Ratio       : 6.24
# ============================================================

# ============================================================
# SYMBOL             : BTCUSD
# LENGHT OF DATA SEARCHED IS : 8352
# Final Balance      : 229162.18
# MODEL TARGET       : ['THL_3H']
# Threshold Used     : 0.514
# Total Trades       : 178
# tp_mult * ATR      : 2.25
# sl_mult * ATR      : 2.25
# BUY                 :  1
# SELL                :  177
# Total Wins         : 95
# Total Losses       : 83
# Win Rate           : 53.37%
# Profit Factor      : 1.11
# Expectancy         : 163.83
# Max Drawdown       : 13.76%
# Sharpe Ratio       : 0.98
# ============================================================

# ============================================================
# SYMBOL             : NZDUSD
# LENGHT OF DATA SEARCHED IS : 6006
# Final Balance      : 252713.90
# MODEL TARGET       : ['THL_3H']
# Threshold Used     : 0.638
# Total Trades       : 57
# tp_mult * ATR      : 2.25
# sl_mult * ATR      : 2.25
# BUY                 :  14
# SELL                :  43
# Total Wins         : 42
# Total Losses       : 15
# Win Rate           : 73.68%
# Profit Factor      : 2.63
# Expectancy         : 924.81
# Max Drawdown       : 2.77%
# Sharpe Ratio       : 8.08
# ============================================================

# ============================================================
# SYMBOL             : AUDJPY
# LENGHT OF DATA SEARCHED IS : 5989
# Final Balance      : 231535.87
# MODEL TARGET       : ['THL_3H']
# Threshold Used     : 0.593
# Total Trades       : 54
# tp_mult * ATR      : 2.25
# sl_mult * ATR      : 2.25
# BUY                 :  20
# SELL                :  34
# Total Wins         : 36
# Total Losses       : 18
# Win Rate           : 66.67%
# Profit Factor      : 1.92
# Expectancy         : 584.00
# Max Drawdown       : 3.28%
# Sharpe Ratio       : 5.11
# ============================================================

# ============================================================
# SYMBOL             : AUDUSD
# LENGHT OF DATA SEARCHED IS : 6000
# Final Balance      : 245451.97
# MODEL TARGET       : ['THL_3H']
# Threshold Used     : 0.530
# Total Trades       : 70
# tp_mult * ATR      : 2.25
# sl_mult * ATR      : 2.25
# BUY                 :  13
# SELL                :  57
# Total Wins         : 48
# Total Losses       : 22
# Win Rate           : 68.57%
# Profit Factor      : 2.04
# Expectancy         : 649.31
# Max Drawdown       : 3.54%
# Sharpe Ratio       : 5.75
# ============================================================





# sl_mult = 1.5

# print(str(sl_mult).replace('.',''))




import pandas as pd
from func import apply_features

SYMBOL = "EURUSD"  
CD_TIME = '10M'

data = pd.read_csv(f'CSV_FILES/MT5_{CD_TIME}_BT_{SYMBOL}_Exchange_Rate_Dataset.csv') 
print(' STARTED APPLY  FEATURES')
df = apply_features(data)
df.dropna(inplace=True)
print('\t DATASET LOADED SUCCESSFULLY ')

