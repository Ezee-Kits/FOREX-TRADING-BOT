

import ta
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from func import apply_features,calc_lot_size,place_buy,check_account_info,place_sell,create_targets,SYMBOL,normalize_lot,get_symbol_volume_info,get_pip_info,log_trade





while True:

    TIMEFRAME = mt5.TIMEFRAME_M5
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

    print(new_df.tail())

    df = apply_features(new_df)
    df.dropna(inplace=True)

    all_target = ['T_5M','T_10M','T_15M','T_20M','T_30M']

    up_moves = {}
    down_moves = {}
    previos_res_df = df[['Open', 'Close']].tail(5).copy()

    for target in all_target:
        print(f'CURRENTLY PREDICTING TARGET : {target}')
        bundle = joblib.load(f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl")
        model = bundle["model"]
        feature_columns = bundle["features"]

        previos_candle = df.loc[:, feature_columns].tail(5)
        previos_pred = model.predict(previos_candle)
        previos_proba = model.predict_proba(previos_candle)
        previos_res_df[target] = previos_pred
        previos_res_df[f'{target}_DN'] = (previos_proba[:, 0] * 100).round(2)
        previos_res_df[f'{target}_UP'] = (previos_proba[:, 1] * 100).round(2)
        next_candle = df.loc[:, feature_columns].tail(1)
        print('NEXT CANDLE DF \n',next_candle)

        assert set(feature_columns) == set(next_candle.columns), "Feature mismatch!"
        y_pred = model.predict(next_candle)
        proba = model.predict_proba(next_candle)
        accu = proba[:,0]*100, proba[:,1]*100
        print(accu)
        res = f'UP MOVEMENT WITH ACCURACY : {round(accu[1][0], 2)}%' if y_pred[0]==1 else f'DOWN MOVEMENT WITH ACCURACY : {round(accu[0][0], 2)}%'
        up_moves[target] = round(accu[1][0], 2)
        down_moves[target] = round(accu[0][0], 2)
        print("✅ PREDICTION FOR NEXT CANDLE:", res)


    print('\n PREVIOUS 5 CANDLES PREDICTION \n',previos_res_df)
    up_moves_mean = round( sum(up_moves.values())/len(up_moves),2) 
    down_moves_mean = round( sum(down_moves.values())/len(down_moves),2)
    print(f'\nAVERAGE UP MOVE ACCURACY : {up_moves_mean}%')
    print(f'AVERAGE DOWN MOVE ACCURACY : {down_moves_mean}%')

    print('===================== ACCOUNT INFORMATIONS ==============================')
    account_info = mt5.account_info()
    balance = account_info.balance
    print("Account Number:", account_info.login)
    print("Balance:", account_info.balance)
    print("Equity:", account_info.equity)
    print("Free Margin:", account_info.margin_free)
    print("Leverage:", account_info.leverage)  

    pip_info = get_pip_info(mt5, SYMBOL)
    pip_size = pip_info["pip_size"]
    pip_value_per_lot = pip_info["pip_value_per_lot"]
    risk_percent = 1  # risk per trade in percentage 

    min_SL_pips = 5      # minimum SL in pips
    max_SL_pips = 200    # maximum SL in pips
    min_TP_pips = 10     # minimum TP in pips
    max_TP_pips = 400    # maximum TP in pips

    # Get current tick
    tick = mt5.symbol_info_tick(SYMBOL)
    ask_price = tick.ask
    bid_price = tick.bid
    spread = ask_price - bid_price  # real-time spread

    # print('pip_size:',pip_size)
    # print('pip_value_per_lot:',pip_value_per_lot)
    # print('ask_price:',ask_price)
    # print('bid_price:',bid_price)
    # print('spread:',spread)

    row = df.iloc[-1]
    ATR_pips = row["ATR"] / pip_size
    # print("ATR in pips:", ATR_pips)

    # Apply ATR multiplier and enforce min/max limits
    SL_pips = max(min(ATR_pips * 1.5, max_SL_pips), min_SL_pips)
    TP_pips = max(min(ATR_pips * 4.5, max_TP_pips), min_TP_pips)
    # print(f"Calculated SL: {SL_pips} pips, TP: {TP_pips} pips")

    # BUY trade SL/TP
    entry_buy = ask_price
    SL_buy = (entry_buy - (SL_pips * pip_size))
    TP_buy = (entry_buy + (TP_pips * pip_size))

    # SELL trade SL/TP
    entry_sell = bid_price
    SL_sell = (entry_sell + (SL_pips * pip_size))
    TP_sell = (entry_sell - (TP_pips * pip_size))

    print(f"BUY Price: {entry_buy}, SL: {SL_buy}, TP: {TP_buy}")
    print(f"SELL Price: {entry_sell}, SL: {SL_sell}, TP: {TP_sell}")
    print('\n=======================================================================')

    # Lot size
    lot_size = calc_lot_size(balance,risk_percent,SL_pips,pip_value_per_lot=pip_value_per_lot,min_lot=0.01,max_lot=2)
    # print("LOT SIZE IS:", lot_size)   


    vol_info = get_symbol_volume_info(mt5, SYMBOL)
    # print("📌 Symbol volume rules:", vol_info)

    lot_size = normalize_lot(lot_size,vol_info["min"],vol_info["max"],vol_info["step"])
    # print("✅ Normalized lot size:", lot_size)


    THRESHOLD = 55
    user = int(input('\n DO YOU WANT TO MAKE A TRADE YES ENTER [1] (((OR))) NO ENTER [2] : '))
    if user == 1:
        if up_moves_mean >= THRESHOLD and up_moves_mean > down_moves_mean:
            # BUY
            print("PLACING BUY ORDER.............")
            result = place_buy(mt5,SYMBOL, lot_size, entry_buy, SL_buy, TP_buy)
            print("BUY ORDER RESULT:", result)
            log_trade(symbol=SYMBOL,direction="BUY",entry_price=entry_buy,SL=SL_buy,
                    TP=TP_buy,lot_size=lot_size,proba_up=up_moves_mean,proba_down=down_moves_mean,order_result=result)

        elif down_moves_mean >= THRESHOLD and down_moves_mean > up_moves_mean:
            # SELL
            print("PLACING BUY ORDER................")
            result = place_sell(mt5,SYMBOL, lot_size, entry_sell , SL_sell, TP_sell)
            print("SELL ORDER RESULT:", result)
            log_trade(symbol=SYMBOL,direction="BUY",entry_price=entry_buy,SL=SL_buy,
                    TP=TP_buy,lot_size=lot_size,proba_up=up_moves_mean,proba_down=down_moves_mean,order_result=result)
        else:
            print("No high-confidence trade")

    else:
        print("NO TRADE EXECUTED AS PER USER REQUEST")