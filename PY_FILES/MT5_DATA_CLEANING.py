

import pandas as pd

# =========================
# 1. LOAD DATA
# =========================

# "EURUSD", "USDJPY", "GBPUSD", "XAUUSD", 'AUDCAD', 'NZDUSD', 'AUDUSD', 'USDCAD',
#  'USDCHF', 'XAGUSD',AUDJPY, CADJPY, EURJPY, GBPJPY, USDSEK, NZDCAD, ETHUSD, BTCUSD

file_path = 'CSV_FILES/XAUUSDm_M10_202605010000_202607081900.csv'

output_file = 'CSV_FILES/MT5_10M_BT_XAUUSD_Exchange_Rate_Dataset.csv'


data = pd.read_csv(
    file_path,
    sep='\t'
)

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
data.columns = data.columns.str.strip()

# =========================
# 3. CREATE DATETIME COLUMN
# =========================
data['Date'] = pd.to_datetime(data['<DATE>'] + ' ' + data['<TIME>'])

# =========================
# 4. RENAME COLUMNS
# =========================
# data = data.rename(columns={
#     '<OPEN>': 'Open',
#     '<HIGH>': 'High',
#     '<LOW>': 'Low',
#     '<CLOSE>': 'Close',
#     '<TICKVOL>': 'TickVol',
#     '<VOL>': 'Volume',
#     '<SPREAD>': 'Spread'
# })

data = data.rename(columns={
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume',
    '<SPREAD>': 'Spread'
})


# # =========================
# # 5. DROP UNUSED COLUMN
# # =========================
data = data.drop(columns=['<VOL>'])   # ❌ Remove useless volume column

# =========================
# 6. SELECT & ORDER COLUMNS
# =========================
data = data[
    # ['Date', 'Open', 'High', 'Low', 'Close','TickVol','Volume', 'Spread']
    ['Date', 'Open', 'High', 'Low', 'Close','Volume', 'Spread']
]

# =========================
# 7. CLEAN DATA
# =========================
data = data.sort_values('Date')
data = data.drop_duplicates()
data = data.reset_index(drop=True)

# =========================
# 8. SAVE CLEAN DATASET
# =========================


data.to_csv(output_file, index=False)

print("[[GOOD]] Clean dataset created successfully!")
print(f"[[MAP]] Saved to: {output_file}")

# Preview
print("\n [[SEARCH]] Preview:")
print(data.head())

