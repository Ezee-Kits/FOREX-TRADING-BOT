

import pandas as pd
import MetaTrader5 as mt5





# Initialize MT5 once
if not mt5.initialize():
    raise RuntimeError("[[BAD]] MT5 initialization failed")
print("[[GOOD]] MT5 initialized successfully")



account_info = mt5.account_info()
balance = account_info.balance

print(f'''
===================== ACCOUNT INFORMATIONS ==============================

Account Number : {account_info.login}
Balance        : {account_info.balance}
''')

["AUDCADm", "AUDUSDm","AUDJPYm", "CADJPYm","EURUSDm",'EURJPYm','GBPJPYm',"NZDUSDm", "USDJPYm","USDCADm"]

TIMEFRAME = mt5.TIMEFRAME_M5
N_BARS = 300

rates = mt5.copy_rates_from_pos('AUDCADm',TIMEFRAME,1,N_BARS )

if rates is None or len(rates) < N_BARS:
    mt5.shutdown()
    raise RuntimeError("[[BAD]] Failed to fetch enough closed candles")

data = pd.DataFrame(rates)
data['Date'] = pd.to_datetime(data['time'], unit='s')
print(data)

data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume',
    'spread': 'Spread'
}, inplace=True)

new_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']]
    
print(new_df)