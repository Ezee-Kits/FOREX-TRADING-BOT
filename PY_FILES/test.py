
#MSE: 0.000369, MAE: 0.015496

# # -----------------------------
# # Plotting
# # -----------------------------
# add_plots = [
#     # EMAs
#     mpf.make_addplot(df["EMA_20"], color="orange"),
#     mpf.make_addplot(df["EMA_50"], color="blue"),
#     mpf.make_addplot(df["EMA_200"], color="red"),

#     # Bollinger Bands
#     mpf.make_addplot(df["BB_H"], color="grey"),
#     mpf.make_addplot(df["BB_L"], color="grey"),

#     # VWAP
#     mpf.make_addplot(df["VWAP"], color="purple"),

#     # RSI & Stochastic
#     mpf.make_addplot(df["RSI"], panel=1, color="green", ylabel="RSI"),
#     mpf.make_addplot(df["STOCH"], panel=1, color="brown"),

#     # MACD
#     mpf.make_addplot(df["MACD"], panel=2, color="black", ylabel="MACD"),

#     # ATR
#     mpf.make_addplot(df["ATR"], panel=3, color="darkred", ylabel="ATR")
# ]

# mpf.plot(
#     df,
#     type="candle",
#     style="yahoo",
#     title="Trading Chart (Trend, Momentum, Volatility)",
#     volume=True,
#     addplot=add_plots,
#     panel_ratios=(6,2,2,2),
#     figsize=(16,10)
# )

# print(48338*0.85)




# info = mt5.symbol_info(SYMBOL)
# print("Filling modes supported:")
# print("FOK:", bool(info.filling_mode & mt5.ORDER_FILLING_FOK))
# print("IOC:", bool(info.filling_mode & mt5.ORDER_FILLING_IOC))
# print("RETURN:", bool(info.filling_mode & mt5.ORDER_FILLING_RETURN))


# import pandas as pd
# import numpy as np
# from func import create_targets,SYMBOL,apply_features

# backtest_data = pd.read_csv(f"CSV_FILES/MT5_5M_BT_{SYMBOL}_Dataset.csv")
# backtest_df = apply_features(backtest_data)
# backtest_df.dropna(inplace=True)
# print(backtest_df['ATR'])

# data = {
#     "Close": [1.1000, 1.1005, 1.1002, 1.1008, 1.1006, 1.1010, 1.1009, 1.1012]
# }
# df = pd.DataFrame(data)
# print(df)

# df_targets = create_targets(df)
# print(df_targets)
# print(SYMBOL)


# a = 7,8
# print(a[0])


# AVERAGE UP MOVE ACCURACY : 46.93%
# AVERAGE DOWN MOVE ACCURACY : 53.07%
# Account Number: 10008981518
# Balance: 99805.89
# Equity: 100423.89
# Free Margin: 79979.75
# Leverage: 100
# Balance: 99805.89
# Balance: 99805.89
# Equity: 100423.89
# Free Margin: 79979.75
# Leverage: 100
# pip_size: 0.0001
# pip_value_per_lot: 10.0
# ask_price: 1.1844000000000001
# bid_price: 1.18438
# spread: 2.0000000000131024e-05
# ATR in pips: 3.6664814017751093
# Calculated SL: 5.499722102662664 pips, TP: 16.49916630798799 pips
# BUY Price: 1.1844000000000001, SL: 1.183850027789734, TP: 1.186049916630799
# SELL Price: 1.18438, SL: 1.1849299722102662, TP: 1.1827300833692012
# LOT SIZE IS: 2
# 📌 Symbol volume rules: {'min': 0.01, 'max': 500.0, 'step': 0.01}
# ✅ Normalized lot size: 2.0
# PS C:\Users\HP\Desktop\PYTHON FILES\#PYTHON\FOREX TRADING>

def a():
    print('a')
    def b():
        print('b')
    return b  # Return the function without parentheses

my_function = a()  # Prints 'a' and stores 'b' in my_function
my_function()      # Now this prints 'b'

        
# def a():
#     print('a')
#     def b():
#         print('b')
#     return b

# main_func = a()
# print(main_func.b())


