import pandas as pd
import os

# ===== CONFIG =====
INPUT_DIR = "crypto_dataset_raw"
OUTPUT_DIR = "crypto_dataset_cleaned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns that should be numeric
NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

def clean_crypto_file(input_path, output_path):
    # Read CSV
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

    # Remove duplicate dates (keep first)
    df = df.drop_duplicates(subset="Date")

    # Sort by date
    df = df.sort_values("Date")

    # Reset index
    df = df.reset_index(drop=True)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned: {os.path.basename(input_path)}")


# ===== RUN FOR ALL FILES =====
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)
        clean_crypto_file(input_path, output_path)
