
import os
import time
import joblib
import MetaTrader5 as mt5
from func import calc_lot_size,place_buy,place_sell,rollover_sleep,move_sl_and_partial_close,get_pip_info,log_trade,Entry_Filtering,Data_Prediction,set_logger,correlation_check,spread_filter,normalize_symbol,forex_market_open


# Initialize MT5 once
if not mt5.initialize():
    raise RuntimeError("[[BAD]] MT5 initialization failed")
print("[[GOOD]] MT5 initialized successfully")


def load_models(model_type, symbol, BASE_PATH, HIGH_TARGETS, CD_TIME):
    symbol = normalize_symbol(symbol = symbol)
    models_dict = {}

    for target in HIGH_TARGETS:
        # file_path = f"{BASE_PATH}/HL_{model_type}_{target}_{symbol}_model.pkl"
        file_path = os.path.join(BASE_PATH,f"HL_{model_type}_{target}_{CD_TIME}_{symbol}_model.pkl")
        

        if not os.path.exists(file_path):
            print(f"Model Path not found: {file_path}")
            raise FileNotFoundError(f"Model not found: {file_path}")

        bundle = joblib.load(file_path)

        models_dict[target] = {
            "model": bundle["model"],
            "features": bundle["features"]
        }

        print(f"[[GOOD]] Loaded {model_type} model for {target}")

    return models_dict




def FOREX_TRADING(SYMBOL, logger):
    set_logger(logger)
    logger.info("BOT STARTED")

    logger.info(f'[[[[[  THIS FUNCTION IS FOR : {SYMBOL} ]]]]')

    BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"ALL_MODELS")

    HIGH_TARGETS = ['THL_3H']

    CD_TIME = '10M'

    # LOAD ALL MODELS HERE
    HELPER_MODELS = load_models(model_type = "MAIN", CD_TIME = CD_TIME,symbol = SYMBOL, BASE_PATH=BASE_PATH, HIGH_TARGETS=HIGH_TARGETS)

    risk_percent = 1

    if 'AUDCAD' in SYMBOL :
        W_threshold = 0.656

    elif 'AUDJPY' in SYMBOL:
        W_threshold = 0.593

    elif 'AUDUSD' in SYMBOL:
        W_threshold = 0.53

    elif 'CADJPY' in SYMBOL :
        W_threshold = 0.641

    elif 'EURJPY' in SYMBOL:
        W_threshold = 0.716

    elif 'EURUSD' in SYMBOL:
        W_threshold = 0.50

    elif 'GBPJPY' in SYMBOL:
        W_threshold = 0.617

    elif 'GBPUSD' in SYMBOL:
        W_threshold = 0.50

    elif 'NZDCAD' in SYMBOL:
        W_threshold = 0.765

    elif 'NZDUSD' in SYMBOL:
        W_threshold = 0.638

    elif 'USDJPY' in SYMBOL:
        W_threshold = 0.565

    elif 'USDSEK' in SYMBOL:
        W_threshold = 0.702
    
    else:
        W_threshold = 1.0 # DIRECTION

    PREV_PRED_PROB = ''

    tp_mult = 1.75
    sl_mult = 1.75

    try:
        while True:

            # if not forex_market_open(mt5=mt5, symbol=SYMBOL):
            #     logger.info("[[IGNORE]] Weekend detected from MT5 server time.")
            #     raise SystemExit("Weekend detected")
            
            rollover_sleep(mt5 = mt5,symbol = SYMBOL)

            Data_Prediction_res = Data_Prediction(mt5 = mt5,SYMBOL = SYMBOL, HIGH_TARGETS = HIGH_TARGETS,HELPER_MODELS = HELPER_MODELS)
            weighted_up = list(Data_Prediction_res[0].values())[0]
            weighted_down = list(Data_Prediction_res[1].values())[0]
            logger.info(f'CURRENTLY PERFORMING SYMBOL : {SYMBOL}')

            df = Data_Prediction_res[2]

            pip_info = get_pip_info(mt5, SYMBOL)
            pip_size = pip_info["pip_size"]
            pip_value_per_lot = pip_info["pip_value_per_lot"]

            tick = mt5.symbol_info_tick(SYMBOL)
            ask_price = tick.ask
            bid_price = tick.bid
            spread = ask_price - bid_price  # real-time spread

            symbol_info = mt5.symbol_info(SYMBOL)
            spread_points = spread / symbol_info.point
            max_spread_points = 70   # example threshold

            if spread_points > max_spread_points:
                logger.info(f"[[WARNING]] Spread too high: {spread_points:.2f} points, skipping trade")
                continue
            
            row = df.iloc[-1]

            positions = mt5.positions_get(symbol=SYMBOL)

            if positions is None:
                logger.info("[[WARNING]] Error fetching positions")
                has_open_trade = False
                has_buy = False
                has_sell = False

            elif len(positions) == 0:
                has_open_trade = False
                has_buy = False
                has_sell = False

            else:
                buy_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
                sell_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

                has_buy = len(buy_positions) > 0
                has_sell = len(sell_positions) > 0
                has_open_trade = has_buy or has_sell   # [[GOOD]] FIX

                if has_buy:
                    logger.info("[[WARNING]] BUY exists")

                if has_sell:
                    logger.info("[[WARNING]] SELL exists")
            account_info = mt5.account_info()
            balance = account_info.balance

            logger.info(f'''
            ===================== ACCOUNT INFORMATIONS ==============================

            SYMBOL : {SYMBOL}

            Account Number : {account_info.login}
            Balance        : {account_info.balance}
            Equity         : {account_info.equity}
            Free Margin    : {account_info.margin_free}
            Leverage       : {account_info.leverage}

            pip_size           : {pip_size}
            pip_value_per_lot  : {pip_value_per_lot}

            tp_mult   : {tp_mult}
            sl_mult   : {sl_mult}

            ask_price : {ask_price}
            bid_price : {bid_price}
            spread    : {spread}

            WEIGHTED UP   : {round(weighted_up * 100, 2)}%
            WEIGHTED DOWN : {round(weighted_down * 100, 2)}%

            THRESHOLD NEEDED : {round(W_threshold * 100, 2)}%

            ========================================================================
            ''')


            if not has_open_trade:
                ##================================= MAIN DIRECTION TRADING ==================================================
                if weighted_up >= W_threshold :

                    logger.info('CHECKING FOR BUY ENTRY LOGIC')

                    corr_allowed = correlation_check(mt5=mt5,new_symbol=SYMBOL,new_trade_type="BUY")

                    if corr_allowed:

                        logger.info(">>>>>>>>> BUY CORRELATION PASSED <<<<<<<<<< ")

                        signal_entry_buy = Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
                            direction="BUY",atr_value=row["ATR"])

                        if signal_entry_buy is None:
                            logger.info("[[BAD]] BUY skipped (no confirmation)")

                        else:
                            PREV_PRED_PROB = weighted_up
                            logger.info("[[UP]] FINAL SIGNAL: BUY")

                            tick = mt5.symbol_info_tick(SYMBOL)
                            entry_buy = tick.ask   # REAL execution price

                            atr = row["ATR"]
                            SL_distance = atr * sl_mult
                            TP_distance = atr * tp_mult
                            SL_pips = SL_distance/ pip_size 

                            spread_ok = spread_filter(mt5=mt5,SYMBOL=SYMBOL,SL_distance=SL_distance)

                            if not spread_ok:
                                continue


                            SL_buy = entry_buy - SL_distance
                            TP_buy = entry_buy + TP_distance

                            lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
                            logger.info(f'LOTS SIZE USED: {lot_size}')

                            result = place_buy(mt5, SYMBOL, lot_size, entry_buy, SL_buy, TP_buy)
                            logger.info(f"BUY ORDER RESULT: {result} ")
                            log_trade(mt5=mt5,symbol=SYMBOL,direction="BUY",entry_price=entry_buy,SL=SL_buy,TP=TP_buy,lot_size=lot_size,
                                proba_up=weighted_up,proba_down=weighted_down,order_result=result)
                    else:
                        logger.info(F'BUY TRADE BLOCKED DUE TO CURRENCY EXPOSER FOR : {SYMBOL}')


                elif weighted_down >= W_threshold:

                    logger.info('CHECKING FOR SELL ENTRY LOGIC')

                    corr_allowed = correlation_check(mt5=mt5,new_symbol=SYMBOL,new_trade_type="SELL")

                    if corr_allowed:

                        logger.info(">>>>>>>>> SELL CORRELATION PASSED <<<<<<<<<< ")

                        signal_entry_sell = Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
                            direction="SELL",atr_value=row["ATR"])

                        if signal_entry_sell is None:
                            logger.info("[[BAD]] SELL skipped (no confirmation)")
                        else:
                            PREV_PRED_PROB = weighted_down
                            logger.info("[[DOWN]] FINAL SIGNAL: SELL")

                            atr = row["ATR"]
                            SL_distance = atr * sl_mult
                            TP_distance = atr * tp_mult
                            SL_pips = SL_distance/ pip_size 

                            tick = mt5.symbol_info_tick(SYMBOL)
                            entry_sell = tick.bid

                            spread_ok = spread_filter(mt5=mt5,SYMBOL=SYMBOL,SL_distance=SL_distance)

                            if not spread_ok:
                                continue

                            SL_sell = entry_sell + SL_distance
                            TP_sell = entry_sell - TP_distance

                            lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
                            logger.info(f'LOTS SIZE USED: {lot_size}')

                            result = place_sell(mt5, SYMBOL, lot_size, entry_sell, SL_sell, TP_sell)
                            logger.info(f"SELL ORDER RESULT: {result}" )
                            log_trade(mt5=mt5,symbol=SYMBOL,direction="SELL",entry_price=entry_sell,
                                SL=SL_sell,TP=TP_sell,lot_size=lot_size,proba_up=weighted_up,proba_down=weighted_down,order_result=result)
                    else:
                        logger.info(F'BUY TRADE BLOCKED DUE TO CURRENCY EXPOSER FOR : {SYMBOL}')
                    
                else:
                    logger.info("[[LOADING]] FINAL SIGNAL: NO TRADE")


                while True:
                    logger.info(f'CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION FOR {SYMBOL}')
                    atr = row["ATR"]
                    output = move_sl_and_partial_close(mt5 = mt5, SYMBOL = SYMBOL, atr_value = atr,HIGH_TARGETS = HIGH_TARGETS,HELPER_MODELS = HELPER_MODELS,W_threshold = W_threshold)
                    if output == 1:
                        logger.info('TP/SL + PARTIAL CONDITION MET')
                        break
                    time.sleep(2)

                    
            elif has_open_trade:
                while True:
                    logger.info(f'CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION FOR {SYMBOL}')
                    atr = row["ATR"]
                    output = move_sl_and_partial_close(mt5 = mt5, SYMBOL = SYMBOL, atr_value = atr,HIGH_TARGETS = HIGH_TARGETS,HELPER_MODELS = HELPER_MODELS,W_threshold = W_threshold)
                    if output == 1:
                        logger.info('TP/SL + PARTIAL CONDITION MET')
                        break
                    time.sleep(2)


    finally:
        # Shutdown MT5 only once, when the bot stops
        mt5.shutdown()
        logger.info("[[GOOD]] MT5 shutdown successfully")
