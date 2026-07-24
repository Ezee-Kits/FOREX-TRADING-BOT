import MetaTrader5 as mt5
import json
import time
import re


# ==========================================================
# INIT MT5
# ==========================================================
if not mt5.initialize():
    raise Exception("MT5 initialization failed")


# ==========================================================
# YOUR SYMBOL LIST (WITH SUFFIX)
# ==========================================================
SYMBOLS = [
    "EURUSDm", "AUDCADm", "CADJPYm","AUDUSDm","BTCUSDm",
    "AUDJPYm", "GBPUSDm", "USDJPYm","ETHUSDm","GBPJPYm",
    "NZDCADm", "NZDUSDm","USDSEKm", "XAGUSDm", "XAUUSDm",
    "EURJPYm", "USDCADm", "USDCHFm",
]



# ==========================================================
# CLEAN SYMBOL FUNCTION (REMOVE m / suffix)
# ==========================================================
def clean_symbol(symbol):
    # removes everything after letters (EURUSDm → EURUSD)
    return re.match(r"[A-Z]+", symbol).group()


# ==========================================================
# GET SYMBOL INFO
# ==========================================================
def get_symbol_info(symbol):

    info = mt5.symbol_info(symbol)

    digits = info.digits

    # True pip size
    if digits in (3, 5):
        pip_size = info.point * 10
    else:
        pip_size = info.point

    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size

    if tick_value > 0 and tick_size > 0:
        pip_value_per_lot = (tick_value / tick_size) * pip_size
    else:
        pip_value_per_lot = None

    spread = (info.ask - info.bid)

    min_gap = info.trade_stops_level * info.point

    if min_gap <= 0:
        min_gap = pip_size * 5


    vol_info = {
        "min": info.volume_min,
        "max": info.volume_max,
        "step": info.volume_step
    }

    return {
        "pip_size": pip_size,
        "pip_value_per_lot": pip_value_per_lot,
        "spread": spread,
        "min_gap": min_gap,
        "vol_info": vol_info
    }


# ==========================================================
# BUILD FINAL DICTIONARY (CLEAN KEYS HERE)
# ==========================================================
SYMBOL_INFO = {}

for symbol in SYMBOLS:

    data = get_symbol_info(symbol)

    if data is not None:

        key = clean_symbol(symbol)   # <<< IMPORTANT FIX

        SYMBOL_INFO[key] = data

        print(f"[[GOOD]] Saved: {symbol} → {key}")

    time.sleep(0.2)


# ==========================================================
# SAVE FILE
# ==========================================================
with open("CSV_FILES/SYMBOL_INFO.json", "w") as f:
    json.dump(SYMBOL_INFO, f, indent=4)

print("\n [[GOOD]] Saved SYMBOL_INFO.json with CLEAN SYMBOL KEYS")


# ==========================================================
# CLOSE MT5
# ==========================================================
mt5.shutdown()