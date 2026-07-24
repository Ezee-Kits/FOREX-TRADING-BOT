import json
import os
import joblib
import json
import numpy as np
import pandas as pd
from func import apply_features,normalize_symbol


# 'AUDCAD' =1H-59-0.585,AUDJPY = 1H-56-0.63,'AUDUSD' = 2H-76-0.528, BTCUSD = NONE,CADJPY = 2H-64-0.518,
#  ETHUSD = NONE,EURJPY = NONE, "EURUSD" = 1H-57-0.61,GBPJPY = NONE,
#  "GBPUSD"=1H-61-0.63, NZDCAD = 1H-68-0.65, 'NZDUSD' = 1H-67-0.655,'USDCAD'=1H-58-0.62, 'USDCHF' = NONE, 
# "USDJPY"=1H-57-0.56 "USDSEK" =1H-101-59 , 'XAGUSD'=NONE, "XAUUSD"=2H-0.52


SYMBOL = "AUDCAD"  
TRADE_NO = 100
CD_TIME = '10M'

risk_percent = 1

tp_mult = 1.75
sl_mult = 1.75


HIGH_TARGETS = ['THL_2H']
weights = {
'THL_2H'  : 1.0
}


BASE_PATH = "ALL_MODELS"

def load_models(model_type,CD_TIME, symbol):
    symbol = normalize_symbol(symbol = symbol)
    models_dict = {}

    for target in HIGH_TARGETS:
        file_path = f"{BASE_PATH}/HL_{model_type}_{target}_{CD_TIME}_{symbol}_model.pkl"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model not found: {file_path}")

        bundle = joblib.load(file_path)

        models_dict[target] = {
            "model": bundle["model"],
            "features": bundle["features"]
        }

        print(f" \n [[GOOD]] Loaded {model_type} model for {target}")

    return models_dict


# LOAD ALL MODELS HERE
HELPER_MODELS = load_models("MAIN", CD_TIME, SYMBOL)

with open("CSV_FILES/SYMBOL_INFO.json") as f:
    SYMBOL_INFO = json.load(f)

info = SYMBOL_INFO[SYMBOL]
pip_size = info["pip_size"]
pip_value_per_lot = info["pip_value_per_lot"]
spread = info["spread"]
min_gap = info["min_gap"]
vol_info = info["vol_info"]



data = pd.read_csv(f'CSV_FILES/MT5_{CD_TIME}_BT_{SYMBOL}_Exchange_Rate_Dataset.csv') 
print('\t DATASET LOADED SUCCESSFULLY \n STARTING APPLY FEATURES ')

#   ==================  REMOVING HIGH SPREAD TIME   =================
data['Date'] = pd.to_datetime(data['Date'])
BAD_HOURS = [21, 22]
data = data[~data['Date'].dt.hour.isin(BAD_HOURS)].reset_index(drop=True)
print('DATA FILE LENGTH BEFORE FILTERING :', len(data))
#=================================================================

df = apply_features(data)
df.dropna(inplace=True)
print('\t APPLY FEATURE SUCCESSFULLY ')
# df.reset_index(drop=True, inplace=True)


working_df = df[['High','Low','Open','Close','ATR']].copy()

for target in HIGH_TARGETS:

    print(f'\n [[ CURRENTLY PREDICTING TARGET : {target} ]]')

    help_model = HELPER_MODELS[target]["model"]
    help_cols = HELPER_MODELS[target]["features"]

    X_help = df[help_cols]

    # Safety check
    assert set(help_cols) == set(X_help.columns)
    help_proba = help_model.predict_proba(X_help)

    up_prob = help_proba[:,1] 
    down_prob = help_proba[:,0]

    working_df[f'{target}_UP'] = up_prob
    working_df[f'{target}_DN'] = down_prob


weight_list = [weights[t] for t in HIGH_TARGETS]
total_weight = sum(weight_list)

working_df['UP_AVG'] = sum(working_df[f"{t}_UP"] * weights[t] for t in HIGH_TARGETS) / total_weight
working_df['DN_AVG'] = sum(working_df[f"{t}_DN"] * weights[t] for t in HIGH_TARGETS) / total_weight

working_df['ATR_PIPS'] = working_df['ATR']/pip_size
working_df['SL_PIPS'] = working_df['ATR_PIPS']*sl_mult
working_df['TP_PIPS'] = working_df['ATR_PIPS']*tp_mult

working_df['NEXT_ENTRY'] = working_df['Open'].shift(-1)
working_df['ASK_PRICE'] = working_df['NEXT_ENTRY']+ spread
working_df['BID_PRICE'] = working_df['NEXT_ENTRY']

working_df['ENTRY_BUY'] = working_df['ASK_PRICE']
working_df['ENTRY_SELL'] = working_df['BID_PRICE']

# BUY trade SL/TP
working_df['SL_BUY'] = working_df['ENTRY_BUY'] - (working_df['SL_PIPS']*pip_size)
working_df['TP_BUY'] = working_df['ENTRY_BUY'] + (working_df['TP_PIPS']*pip_size)

# SELL trade SL/TP
working_df['SL_SELL'] = working_df['ENTRY_SELL'] + (working_df['SL_PIPS']*pip_size)
working_df['TP_SELL'] = working_df['ENTRY_SELL'] - (working_df['TP_PIPS']*pip_size)



def run_backtest(working_df, threshold):

    signals = np.zeros(len(working_df), dtype=np.int8)

    up = working_df['UP_AVG'].to_numpy()
    dn = working_df['DN_AVG'].to_numpy()

    signals[up >= threshold] = 1
    signals[dn >= threshold] = -1

    highs = working_df['High'].to_numpy()
    lows = working_df['Low'].to_numpy()

    tp_buy = working_df['TP_BUY'].to_numpy()
    sl_buy = working_df['SL_BUY'].to_numpy()

    tp_sell = working_df['TP_SELL'].to_numpy()
    sl_sell = working_df['SL_SELL'].to_numpy()

    wins = 0
    losses = 0

    i = 0
    n = len(signals)

    while i < n - 1:

        signal = signals[i]

        if signal == 0:
            i += 1
            continue

        if signal == 1:

            sl = sl_buy[i]
            tp = tp_buy[i]

            j = i + 1

            while j < n:

                if lows[j] <= sl:
                    losses += 1
                    i = j
                    break

                if highs[j] >= tp:
                    wins += 1
                    i = j
                    break

                j += 1

            if j == n:
                i += 1

        else:

            sl = sl_sell[i]
            tp = tp_sell[i]

            j = i + 1

            while j < n:

                if highs[j] >= sl:
                    losses += 1
                    i = j
                    break

                if lows[j] <= tp:
                    wins += 1
                    i = j
                    break

                j += 1

            if j == n:
                i += 1

    total = wins + losses

    return {
        "THRESHOLD": round(threshold, 3),
        "TRADES": total,
        "WINS": wins,
        "LOSSES": losses,
        "WIN_RATE": round(wins / total, 4) if total else 0
    }


results = []
thresholds = sorted(set(pd.concat([working_df['UP_AVG'],working_df['DN_AVG']]).round(3).dropna()))
print('THRESHOL LENGHT >>',len(thresholds))

for threshold in thresholds:
    print(f'CURRENTLY ON {threshold} THRESHOLD >> {thresholds.index(threshold)+1}/{len(thresholds)}')

    result = run_backtest(
        working_df,
        threshold
    )

    results.append(result)

result_df = pd.DataFrame(results)
result_df = result_df[result_df['TRADES'] >= TRADE_NO]

result_df = result_df.sort_values(by='WIN_RATE',ascending=False)

print(f'SYMBOL: {SYMBOL}')
print(result_df.head(10).to_string(index=False))