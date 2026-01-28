# import MetaTrader5 as mt5
# import pandas as pd
# import datetime

# # Connect
# if not mt5.initialize():
#     print("MT5 failed to initialize")
#     quit()

# symbol = "EURUSD"
# timeframe = mt5.TIMEFRAME_M30

# start = datetime.datetime(2023,1,1)
# end = datetime.datetime(2025,11,30)

# data = mt5.copy_rates_range(symbol, timeframe, start, end)

# df = pd.DataFrame(data)
# df['time'] = pd.to_datetime(df['time'], unit='s')

# print(df.tail())
# df.to_csv('CSV FILES/MT5_EURUSD_Exchange_Rate_Dataset.csv',mode='a', index=False)
# mt5.shutdown()

import MetaTrader5 as mt5
import pandas as pd
import datetime
from func import drop_duplicate,SYMBOL


# Connect to MT5
if not mt5.initialize():
    print("MT5 failed to initialize")
    quit()


timeframe = mt5.TIMEFRAME_M5

overall_start = datetime.datetime(2020, 1, 1)
overall_end = datetime.datetime(2025, 12, 30)
chunk_months = 3
current_start = overall_start

file_path = f'CSV_FILES/MT5_5M_{SYMBOL}_Exchange_Rate_Dataset.csv'

# Create or overwrite the CSV
open(file_path, 'w').close()

while current_start < overall_end:
    # Calculate chunk end
    month = current_start.month + chunk_months
    year = current_start.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = current_start.day
    try:
        current_end = datetime.datetime(year, month, day)
    except:
        current_end = datetime.datetime(year, month, 1) - datetime.timedelta(days=1)
    
    if current_end > overall_end:
        current_end = overall_end

    print(f"Downloading data: {current_start.date()} → {current_end.date()}")
    
    data = mt5.copy_rates_range(SYMBOL, timeframe, current_start, current_end)
    
    if data is not None and len(data) > 0:
        df = pd.DataFrame(data)
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # Rename columns to Yahoo Finance style
        df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        # Keep only necessary columns
        df = df[['Date','Open','High','Low','Close','Volume']]
        
        # Append to CSV with header only for first chunk
        df.to_csv(file_path, mode='a', index=False, header=not pd.io.common.file_exists(file_path))
        print(f"Saved {len(df)} rows")
    else:
        print("No data returned for this chunk")
    
    # Move to next chunk
    current_start = current_end + datetime.timedelta(days=1)

mt5.shutdown()
drop_duplicate(path=file_path)
print("Done downloading all chunks with proper headers!")
