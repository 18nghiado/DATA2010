import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# normalize per asset using first Close
full_df["normalized_price"] = full_df["Close"] / full_df.groupby("asset")["Close"].transform("first")

market_index = (
    full_df.groupby("Date")["normalized_price"]
    .median()          # median keeps it robust
    .reset_index()
)

btc_df = full_df[full_df["asset"] == "bitcoin"].copy()

btc_df["btc_normalized"] = (
    btc_df["Close"] / btc_df["Close"].iloc[0]
)

# Visual 1
plt.figure(figsize=(12,6))

plt.plot(market_index["Date"], market_index["normalized_price"],
         label="Market Index (Median)")

plt.plot(btc_df["Date"], btc_df["btc_normalized"],
         label="Bitcoin")

plt.title("Crypto Market Index vs Bitcoin Price", fontsize=30)

plt.ylabel("Normalized Price", fontsize=28)
plt.xlabel("Date", fontsize=28)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=30)

plt.tight_layout()
plt.savefig("fig1_market_index.png", dpi=300, bbox_inches="tight")
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
ax1.set_ylabel("Normalized Price", fontsize=28)
ax1.set_xlabel("Date", fontsize=28)
ax1.tick_params(axis="x", labelsize=15)
ax1.tick_params(axis="y", labelsize=15)

# Second y-axis for sentiment
ax2 = ax1.twinx()
ax2.plot(
    daily_sentiment["Date"],
    daily_sentiment["sentiment_smooth"],
    color="orange",
    label="Average Sentiment (30d MA)"
)
ax2.set_ylabel("Average Sentiment Score", fontsize=28)
ax2.tick_params(axis="y", labelsize=15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=30)

plt.title("Market Index vs Average Sentiment (30-Day Rolling)", fontsize=30)
plt.tight_layout()
plt.savefig("fig2_market_vs_sentiment.png", dpi=300, bbox_inches="tight")
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
plt.title("30-Day Rolling Volatility of Crypto Market Index", fontsize=30)
plt.ylabel("Volatility", fontsize=28)
plt.xlabel("Date", fontsize=28)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("fig3_rolling_volatility.png", dpi=300, bbox_inches="tight")
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


plt.figure(figsize=(12,6))

order = ["Pre-Crash", "Crash 2022", "Recovery 2023"]
sns.boxplot(x="period", y="iqr", data=dispersion, order=order)

plt.title("Cross-Asset Dispersion by Market Period", fontsize=30)
plt.ylabel("IQR of Normalized Prices", fontsize=28)
plt.xlabel("Market Period", fontsize=28)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("fig4_dispersion_by_period.png", dpi=300, bbox_inches="tight")
plt.show()

