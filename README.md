# DATA2010 — Cryptocurrency Price Prediction

A research project exploring binary classification and regression models for predicting cryptocurrency price movements using historical price data and news sentiment features.

## Overview

This project investigates whether next-day cryptocurrency price direction (up/down) and price value can be predicted using a combination of:
- Historical OHLCV price features (`Open`, `High`, `Low`, `Close`, `Volume`)
- Technical indicators (`ma_7`, `ma_30`, `ma_ratio`, `log_return`, `vol_7d`, `vol_30d`)
- News-derived sentiment features (`news_count`, `sentiment_polarity_mean`, `sentiment_score_mean`, `sentiment_subjectivity_mean`)

Models evaluated include Logistic Regression, Random Forest, Gradient Boosting (GBM), SVM, KNN.

---

## Project Structure

```
DATA2010/
├── data/                                   # Raw and preprocessed CSV datasets
├── src/                                    # Source modules
|   ├── data
│   │  ├── timeseries_data.py               # TimeSeriesDataset dataclass with split utilities
│   │  └── dataloader.py                    # Dataloader class for reading CSV into TimeSeriesDataset
│   ├── utils                               
│   |   └── seed.py                         # Seed for reproduce
│   └─ pipelines
│      ├── cleaning.py                      # Data cleaning script 
│      └── eda.py                           # Exploratory Data Analysis and Visualization script   
├── cleaning.ipynb                          # Interactive data cleaning notebook
├── experiments_standard_pipeline.ipynb     # Baseline sklearn pipeline experiments
├── EDA_Report_Notebook                     # Interactive EDA and Visualization notebook
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/18nghiado/DATA2010.git
cd DATA2010
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

---

## Usage

### Data Cleaning

Run the cleaning script to preprocess raw data:

```bash
python cleaning.py
```

Or interactively via the notebook:

```bash
jupyter notebook cleaning.ipynb
```

### Running Experiments

Open the main experiments notebook:

```bash
jupyter notebook experiments.ipynb
```

For the standard sklearn baseline pipeline:

```bash
jupyter notebook experiments_standard_pipeline.ipynb
```

---

## Contributors

- [Bao Ngo](https://github.com/benjaminnNgo)   - 7951466
- [Duc Nghia Do](https://github.com/18nghiado) - 7934473
