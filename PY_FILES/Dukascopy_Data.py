import requests
import pandas as pd
import struct
from datetime import datetime, timedelta
from tqdm import tqdm

# ================= CONFIG =================
SYMBOL = "EURUSD"
START_DATE = "2018-01-01"
END_DATE   = "2018-01-31"
OUTPUT_CSV = "dukascopy_EURUSD_5M.csv"
# ==========================================

DUKASCOPY_SYMBOL = SYMBOL.lower()
BASE_URL = "https://datafeed.dukascopy.com/datafeed"

def download_m1_day(symbol, date):
    """Download Dukascopy 1-minute candles for a specific day."""
    year  = date.year
    month = date.month - 1  # Dukascopy uses 0-based month
    day   = date.day

    url = f"{BASE_URL}/{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"

    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.content
    candles = []

    # Each candle is 20 bytes
    for i in range(0, len(data), 20):
        chunk = data[i:i+20]
        if len(chunk) < 20:
            continue

        # Unpack 6 integers: timestamp offset, open, high, low, close, volume
        timestamp_offset, open_, high, low, close, volume = struct.unpack(">6I", chunk)

        # Convert prices to float
        open_  /= 100000
        high   /= 100000
        low    /= 100000
        close  /= 100000

        candles.append((open_, high, low, close, volume))

    return candles

def build_m1_dataframe(symbol, start, end):
    all_rows = []
    current = start

    with tqdm(total=(end - start).days + 1) as pbar:
        while current <= end:
            candles = download_m1_day(symbol, current)
            if candles:
                day_start = datetime(current.year, current.month, current.day)
                for i, c in enumerate(candles):
                    timestamp = day_start + timedelta(minutes=i)
                    all_rows.append([timestamp, c[0], c[1], c[2], c[3], c[4]])
            current += timedelta(days=1)
            pbar.update(1)

    df = pd.DataFrame(all_rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime type
    df.set_index("Date", inplace=True)
    return df

def resample_to_m5(df):
    """Resample 1-minute candles to 5-minute candles."""
    return df.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

if __name__ == "__main__":
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(END_DATE, "%Y-%m-%d")

    print("Downloading Dukascopy M1 data...")
    df_m1 = build_m1_dataframe(DUKASCOPY_SYMBOL, start, end)

    print("Resampling to M5...")
    df_m5 = resample_to_m5(df_m1)

    df_m5.reset_index(inplace=True)
    df_m5.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Saved {len(df_m5)} rows to {OUTPUT_CSV}")
