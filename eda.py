import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CLEAN_DIR = "data/clean"

all_dfs = []
for file in os.listdir(CLEAN_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(CLEAN_DIR, file))
        df["asset"] = file.replace(".csv", "")
        all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)

full_df["Date"] = pd.to_datetime(full_df["Date"])
full_df = full_df.sort_values(["asset", "Date"])

# normalize per asset using first Close (same start date across assets)
full_df["normalized_price"] = full_df["Close"] / full_df.groupby("asset")["Close"].transform("first")

market_index = (
    full_df.groupby("Date")["normalized_price"]
    .median()          # median keeps it robust
    .reset_index()
)

# Visual 1
plt.figure(figsize=(12,6))
plt.plot(market_index["Date"], market_index["normalized_price"])
plt.title("Crypto Market Index (Median of Normalized Prices)")
plt.ylabel("Normalized Price (relative to 2021-10-12)")
plt.xlabel("Date")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()

# Visual 2

# Compute daily average sentiment across assets
daily_sentiment = (
    full_df.groupby("Date")["sentiment_score_mean"]
    .mean()
    .reset_index()
)

# Smooth both series
market_index_smooth = market_index.copy()
market_index_smooth["normalized_price_smooth"] = (
    market_index_smooth["normalized_price"].rolling(30).mean()
)

daily_sentiment["sentiment_smooth"] = (
    daily_sentiment["sentiment_score_mean"].rolling(30).mean()
)

fig, ax1 = plt.subplots(figsize=(12,6))

# Plot Market Index
ax1.plot(
    market_index_smooth["Date"],
    market_index_smooth["normalized_price_smooth"],
    label="Market Index (30d MA)"
)
ax1.set_ylabel("Normalized Price")

# Second y-axis for sentiment
ax2 = ax1.twinx()
ax2.plot(
    daily_sentiment["Date"],
    daily_sentiment["sentiment_smooth"],
    color="orange",
    label="Average Sentiment (30d MA)"
)
ax2.set_ylabel("Average Sentiment Score")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Market Index vs Average Sentiment (30-Day Rolling)")
plt.tight_layout()
plt.show()

# Visual 3

market_index["log_return"] = np.log(
    market_index["normalized_price"] /
    market_index["normalized_price"].shift(1)
)

market_index["vol_30d"] = market_index["log_return"].rolling(30).std()

plt.figure(figsize=(12,6))
vol_df = market_index.dropna(subset=["vol_30d"])
plt.plot(vol_df["Date"], vol_df["vol_30d"])
plt.title("30-Day Rolling Volatility of Crypto Market Index")
plt.ylabel("Volatility")
plt.show()

# Visual 4
dispersion = (
    full_df.groupby("Date")["normalized_price"]
    .agg(iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
    .reset_index()
)

dispersion["period"] = "Other"
dispersion.loc[dispersion["Date"] < "2022-04-01", "period"] = "Pre-Crash"
dispersion.loc[
    (dispersion["Date"] >= "2022-04-01") &
    (dispersion["Date"] < "2023-01-01"),
    "period"
] = "Crash 2022"
dispersion.loc[dispersion["Date"] >= "2023-01-01", "period"] = "Recovery 2023"


plt.figure(figsize=(8,6))
import seaborn as sns
order = ["Pre-Crash", "Crash 2022", "Recovery 2023"]
sns.boxplot(x="period", y="iqr", data=dispersion, order=order)

plt.title("Cross-Asset Dispersion by Market Period")
plt.ylabel("IQR of Normalized Prices")
plt.xlabel("Market Period")
plt.tight_layout()
plt.show()


plt.savefig("fig1_market_index.png", dpi=300, bbox_inches="tight")
plt.savefig("fig2_market_vs_sentiment.png", dpi=300, bbox_inches="tight")
plt.savefig("fig3_rolling_volatility.png", dpi=300, bbox_inches="tight")
plt.savefig("fig4_dispersion_by_period.png", dpi=300, bbox_inches="tight")
