
import json
import os
import joblib
import json
import numpy as np
import pandas as pd
from func import apply_features,normalize_lot,normalize_symbol #,SYMBOL_INFO


# 'AUDCAD' =1H-59-0.585,AUDJPY = 1H-56-0.63,'AUDUSD' = 2H-76-0.528, BTCUSD = NONE,CADJPY = 2H-64-0.518, ETHUSD = NONE,EURJPY = NONE, 
#  "EURUSD" = 1H-57-0.61,GBPJPY = NONE,
#  "GBPUSD"=1H-61-0.63, NZDCAD = 1H-68-0.65, 'NZDUSD' = 1H-67-0.655,'USDCAD'=1H-58-0.62, 'USDCHF' = NONE, "USDJPY"=1H-57-0.56
#  "USDSEK" =1H-101-59 , 'XAGUSD'=NONE, "XAUUSD"=2H-0.52



SYMBOL = "AUDCAD"  
HIGH_TARGETS = ['THL_3H']
CD_TIME = '10M'


weights = {
'THL_3H'  : 1.0
}

risk_percent = 1
W_threshold = 0.531 # DIRECTION 

tp_mult = 1.75
sl_mult = 1.75

BASE_PATH = "ALL_MODELS"

def load_models(model_type, CD_TIME,symbol):
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




data = pd.read_csv(f'CSV_FILES/MT5_10M_BT_{SYMBOL}_Exchange_Rate_Dataset.csv') 

#   ==================  REMOVING HIGH SPREAD TIME   =================
data['Date'] = pd.to_datetime(data['Date'])
BAD_HOURS = [21, 22]
data = data[~data['Date'].dt.hour.isin(BAD_HOURS)].reset_index(drop=True)
print('DATA FILE LENGTH BEFORE FILTERING :', len(data))
#=================================================================

df = apply_features(data)
df.dropna(inplace=True)
print('\t DATASET LOADED SUCCESSFULLY ')
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

def calc_lot_size(
        balance,risk_percent,sl_pips,pip_value_per_lot,min_lot,max_lot):

    risk_amount = balance * (risk_percent / 100)

    lot_cal = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(min_lot, min(lot_cal, max_lot))

    return lot


def normalize_lot(lot, vol_min, vol_max, vol_step):

    lot = max(vol_min, min(lot, vol_max))

    lot = np.floor(lot / vol_step) * vol_step

    return round(lot, 2)

    
working_df['SIGNAL'] = 0  # 1 = BUY, -1 = SELL
working_df.loc[(working_df['UP_AVG'] >= W_threshold),'SIGNAL'] = 1

working_df.loc[(working_df['DN_AVG'] >= W_threshold),'SIGNAL'] = -1


balance = 200000
No_trades = 0
win_count = 0
loss_count = 0
total_gross_profit = 0
total_gross_loss = 0
history = []
equity_curve = []


i = 0

while i < len(working_df) - 1:
    # working_df['LOTS'] = calc_lot_size(balance,risk_percent,working_df['SL_PIPS'],pip_value_per_lot=pip_value_per_lot,min_lot=0.01,max_lot=2)
    # working_df['LOT_SIZE'] = normalize_lot(working_df['LOTS'],vol_info["min"],vol_info["max"],vol_info["step"])

    row = working_df.iloc[i]

    sl_pips = row['SL_PIPS']

    lot_size = calc_lot_size(balance,risk_percent,sl_pips,pip_value_per_lot,min_lot=0.01,max_lot=50)

    lot_size = normalize_lot(lot_size,vol_info["min"],vol_info["max"],vol_info["step"])

    if row['SIGNAL'] == 0:
        i += 1
        continue

    No_trades += 1

    entry_buy = row['ENTRY_BUY']
    entry_sell = row['ENTRY_SELL']

    SL_buy = row['SL_BUY']
    TP_buy = row['TP_BUY']

    SL_sell = row['SL_SELL']
    TP_sell = row['TP_SELL']

    # lot_size = row['LOT_SIZE']

    weighted_up = row['UP_AVG']
    weighted_down = row['DN_AVG']
    trade_date = working_df.index[i]

    trade_taken = False

    for j in range(i+1, len(working_df)):

        high = working_df.iloc[j]['High']
        low = working_df.iloc[j]['Low']

        # ================= BUY =================
        if row['SIGNAL'] == 1:

            if low <= SL_buy:
                loss = ((entry_buy - SL_buy)/pip_size) * pip_value_per_lot * lot_size
                balance -= loss
                total_gross_loss += loss
                loss_count += 1
                equity_curve.append(balance)
                history.append({'DATE': trade_date, 'DIRECTION': 'BUY', 'RESULT': 'LOSS', 'WEIGHTED_UP': weighted_up})
                i = j
                trade_taken = True
                break

            elif high >= TP_buy:
                profit = ((TP_buy - entry_buy)/pip_size) * pip_value_per_lot * lot_size
                balance += profit
                total_gross_profit += profit
                win_count += 1
                equity_curve.append(balance)
                history.append({'DATE': trade_date, 'DIRECTION': 'BUY', 'RESULT': 'WIN', 'WEIGHTED_UP': weighted_up})
                i = j
                trade_taken = True
                break

        # ================= SELL =================
        elif row['SIGNAL'] == -1:

            if high >= SL_sell:
                loss = ((SL_sell - entry_sell)/pip_size) * pip_value_per_lot * lot_size
                balance -= loss
                total_gross_loss += loss
                loss_count += 1
                equity_curve.append(balance)
                history.append({'DATE': trade_date, 'DIRECTION': 'SELL', 'RESULT': 'LOSS', 'WEIGHTED_DOWN': weighted_down})
                i = j
                trade_taken = True
                break

            elif low <= TP_sell:
                profit = ((entry_sell - TP_sell)/pip_size) * pip_value_per_lot * lot_size
                balance += profit
                total_gross_profit += profit
                win_count += 1
                equity_curve.append(balance)
                history.append({'DATE': trade_date, 'DIRECTION': 'SELL', 'RESULT': 'WIN', 'WEIGHTED_DOWN': weighted_down})
                i = j
                trade_taken = True
                break





    if not trade_taken:
        i += 1



df = pd.DataFrame(history)
print(df.to_string())




# =====================================================
# MAXIMUM DRAWDOWN
# =====================================================

equity_series = pd.Series(equity_curve)

running_peak = equity_series.cummax()

drawdowns = (running_peak - equity_series) / running_peak

max_drawdown = drawdowns.max()

# =====================================================
# WIN RATE
# =====================================================

total_trades = win_count + loss_count

if total_trades > 0:
    win_rate = (win_count / total_trades) * 100
else:
    win_rate = 0

# =====================================================
# PROFIT FACTOR
# =====================================================

if total_gross_loss > 0:
    profit_factor = total_gross_profit / total_gross_loss
else:
    profit_factor = float("inf")

# =====================================================
# EXPECTANCY
# =====================================================

if win_count > 0:
    avg_win = total_gross_profit / win_count
else:
    avg_win = 0

if loss_count > 0:
    avg_loss = total_gross_loss / loss_count
else:
    avg_loss = 0

win_probability = win_count / total_trades if total_trades > 0 else 0
loss_probability = loss_count / total_trades if total_trades > 0 else 0

expectancy = (
    (win_probability * avg_win)
    -
    (loss_probability * avg_loss)
)

# =====================================================
# SHARPE RATIO
# =====================================================

returns = equity_series.pct_change().dropna()

if len(returns) > 1 and returns.std() > 0:

    sharpe_ratio = (
        returns.mean()
        /
        returns.std()
    ) * np.sqrt(252)

else:

    sharpe_ratio = 0



# =====================================================
# PRINT RESULTS
# =====================================================

print("\n" + "="*60)

print(f"SYMBOL             : {SYMBOL}")
print(f'LENGHT OF DATA SEARCHED IS : {len(data)}')
print(f"Final Balance      : {balance:.2f}")
print(f"MODEL TARGET       : {HIGH_TARGETS}")

print(f"Threshold Used     : {W_threshold:.3f}")
print(f"Total Trades       : {total_trades}")

print(f"tp_mult * ATR      : {tp_mult}")
print(f"sl_mult * ATR      : {sl_mult}")

print('BUY                 : ', (df['DIRECTION'] == 'BUY').sum())
print('SELL                : ', (df['DIRECTION'] == 'SELL').sum())

print(f"Total Wins         : {win_count}")
print(f"Total Losses       : {loss_count}")

print(f"Win Rate           : {win_rate:.2f}%")

print(f"Profit Factor      : {profit_factor:.2f}")

print(f"Expectancy         : {expectancy:.2f}")

print(f"Max Drawdown       : {max_drawdown:.2%}")

print(f"Sharpe Ratio       : {sharpe_ratio:.2f}")

print("="*60)
