import pandas as pd
import numpy as np
import os

# ===== CONFIG =====
INPUT_DIR = "crypto_dataset_raw"
OUTPUT_DIR = "crypto_dataset_cleaned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns that should be numeric
NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

def clean_crypto_file(input_path, output_path):
    asset = os.path.splitext(os.path.basename(input_path))[0].upper()
    df = pd.read_csv(input_path)

    # Drop rows where Date is missing or not a real date
    df = df[df["Date"].notna()]

    # Convert Date to datetime (invalid ones become NaT)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where all price columns are missing
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")

    # ---------- Remove duplicates & sort ----------
    df = df.drop_duplicates(subset="Date")
    df = df.sort_values("Date").reset_index(drop=True)

    # ---------- OHLC consistency ----------
    bad_ohlc = (
        (df["Low"] > df["High"]) |
        (df["Open"] < df["Low"]) | (df["Open"] > df["High"]) |
        (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
    )
    df = df[~bad_ohlc]

    # ---------- Remove zero-liquidity rows ----------
    df = df[~((df["Volume"] == 0) & (df["Open"] == df["Close"]))]

    # ---------- Feature engineering ----------
    #df["asset"] = asset

    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["vol_7d"] = df["log_return"].rolling(7).std()
    df["vol_30d"] = df["log_return"].rolling(30).std()

    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["ma_ratio"] = df["ma_7"] / df["ma_30"]

    # ---------- Final tidy ----------
    df = df.reset_index(drop=True)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned: {asset}")


# ===== RUN FOR ALL FILES =====
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)
        clean_crypto_file(input_path, output_path)
