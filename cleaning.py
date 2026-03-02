import pandas as pd
import numpy as np
import os
import json

# ===== CONFIG =====
DATA_DIR = "data"

RAW_DIR = os.path.join(DATA_DIR, "raw")
TOKEN_DIR = os.path.join(RAW_DIR, "token_datasets")
SEMANTIC_DIR = os.path.join(RAW_DIR, "semantic")

OUTPUT_DIR = os.path.join(DATA_DIR, "clean")

NEWS_FILEPATH = os.path.join(SEMANTIC_DIR, "cryptonews.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns that should be numeric
NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

def load_daily_sentiment(filepath):

    news = pd.read_csv(filepath)
    news["Date"] = pd.to_datetime(news["Date"], errors="coerce")
    news = news.dropna(subset=["Date"])

    news["Date"] = news["Date"].dt.floor("D")

    # Parse JSON safely
    def parse_sentiment(x):
        try:
            if isinstance(x, str):
                return json.loads(x.replace("'", '"'))
            return {}
        except:
            return {}

    sentiment = news["Sentiment"].apply(parse_sentiment).apply(pd.Series)

    # Convert class to numeric score
    class_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    sentiment["score"] = sentiment["class"].map(class_map)

    sentiment["polarity"] = pd.to_numeric(sentiment["polarity"], errors="coerce")
    sentiment["subjectivity"] = pd.to_numeric(sentiment["subjectivity"], errors="coerce")

    expanded = pd.concat([news[["Date", "text"]], sentiment], axis=1)

    # Aggregate per day
    daily = (
        expanded.groupby("Date")
        .agg(
            sentiment_score_mean=("score", "mean"),
            sentiment_polarity_mean=("polarity", "mean"),
            sentiment_subjectivity_mean=("subjectivity", "mean"),
            news_count=("score", "count"),
            news_text=("text", lambda x: ".".join(x.dropna()))
        )
        .reset_index()
    )

    return daily


DAILY_SENTIMENT = load_daily_sentiment(NEWS_FILEPATH)

FIRST_NEWS_DATE = DAILY_SENTIMENT["Date"].min()
LAST_NEWS_DATE = DAILY_SENTIMENT["Date"].max()

SENTIMENT_COLS = [
    "sentiment_score_mean",
    "sentiment_polarity_mean",
    "sentiment_subjectivity_mean",
    "news_count"
]

def clean_crypto_file(input_path, output_path):
    asset = os.path.splitext(os.path.basename(input_path))[0].upper()
    df = pd.read_csv(input_path)

    # Convert blank / whitespace cells to NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Drop rows where Date is missing or not a real date
    df = df[df["Date"].notna()]

    # Convert Date to datetime (invalid ones become NaT)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        
    for col in NUMERIC_COLS:
        # Convert empty strings or invalid to NaN
        df[col] = df[col].replace("", np.nan)

        # Fill NaN with median of that column
        median_value = df[col].median()

        df[col] = df[col].fillna(median_value)

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

    # ---------- Merge sentiment ----------
    df = df.merge(DAILY_SENTIMENT, on="Date", how="inner")

    # ---------- Feature engineering ----------
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["vol_7d"] = df["log_return"].rolling(7).std()
    df["vol_30d"] = df["log_return"].rolling(30).std()

    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["ma_ratio"] = df["ma_7"] / df["ma_30"]

    df["return"] = df["return"].fillna(0)
    df["log_return"] = df["log_return"].fillna(0)

    df["vol_7d"] = df["vol_7d"].fillna(0)
    df["vol_30d"] = df["vol_30d"].fillna(0)

    df["ma_7"] = df["ma_7"].fillna(0)
    df["ma_30"] = df["ma_30"].fillna(0)

    df["ma_ratio"] = df["ma_ratio"].fillna(0)


    # ---------- Labels  ----------
    df["next_close"] = df["Close"].shift(-1)

    # 1 if next day's close is higher than today's close, else 0
    df["price_increase"] = (df["next_close"] > df["Close"]).astype("Int64")

    # Set last row to nan explicitly (since next_close is NA)
    df.loc[df["next_close"].isna(), "price_increase"] = np.nan

    df = df.reset_index(drop=True)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned: {asset}")


# ===== RUN FOR ALL FILES =====
for file in os.listdir(TOKEN_DIR):

    if file.endswith(".csv"):

        input_path = os.path.join(TOKEN_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)

        clean_crypto_file(input_path, output_path)
