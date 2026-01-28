import yfinance as yf







pair = "EURUSD=X"   # Symbol format must include =X
df = yf.download(pair, start="2023-01-01", end="2025-11-30",interval="30m")

print(df.head())

df.to_csv('CSV FILES/EURUSD_Exchange_Rate_Dataset.csv')
