📈 FOREX TRADING (Machine Learning Based Trading System)
Overview

This project is a machine learning–based Forex trading system built to analyze historical market data, train predictive models, backtest trading strategies, and generate next-candle trade predictions.

The system focuses on EURUSD (and some XAUUSD datasets) and supports multiple timeframes such as 5M, 10M, 15M, 20M, and 30M.

The goal of this project is educational, experimental, and research-oriented, allowing traders and developers to understand how machine learning can be applied to Forex market prediction and strategy evaluation.

⚠️ This project does NOT guarantee profit. Forex trading involves significant risk.


#Key Capabilities
  Machine learning price direction prediction (UP / DOWN)
  
  Multiple timeframe support (5M → 30M)
  
  Model loading and inference
  
  Backtesting on historical data
  
  MT5 and external data support
  
  Automated preprocessing & feature engineering
  
  Trade logging and performance analysis


#Project Structure (Detailed Explanation)


FOREX TRADING/
│
├── ALL_MODELS/
│   ├── EURUSD_lgbm_bundle.pkl
│   ├── EURUSD_lgbm_T_5M.pkl
│   ├── EURUSD_lgbm_T_10M.pkl
│   ├── EURUSD_lgbm_T_15M.pkl
│   ├── EURUSD_lgbm_T_20M.pkl
│   ├── EURUSD_lgbm_T_30M.pkl

├── CSV_FILES/
│   ├── BACKTEST_DATA.csv
│   ├── dukascopy_EURUSD_5M.csv
│   ├── EURUSD_Exchange_Rate_Dataset.csv
│   ├── MT5_5M_BT_EURUSD_Dataset.csv
│   ├── MT5_5M_EURUSD_Exchange_Rate_Dataset.csv
│   ├── MT5_5M_XAUUSD_Exchange_Rate_Dataset.csv
│   ├── MT5_10M_EURUSD_Exchange_Rate_Dataset.csv
│   ├── MT5_EURUSD_Exchange_Rate_Dataset.csv
│   ├── Trade_log.csv

├── PY_FILES/
│   ├── ALL_BACKTEST.py
│   ├── ALL_PRED_NXT.py
│   ├── ALL_PROCESS.py
│   ├── Dukascopy_Data.py
│   ├── func.py
│   ├── Get_dataMT5.py
│   ├── Get_dataYF.py
│   ├── Load_model.py
│   ├── PRED_NEXT.py
│   ├── Preprocessing.py
│   ├── test.py

└── requirements.txt

Folder Explanation
  1️⃣ ALL_MODELS
    This folder contains pre-trained machine learning models saved using joblib or pickle.
    
    Each model is trained for a specific timeframe:
    
    EURUSD_lgbm_T_5M.pkl → 5-minute timeframe
    
    EURUSD_lgbm_T_10M.pkl → 10-minute timeframe
    
    EURUSD_lgbm_T_15M.pkl → 15-minute timeframe
    
    EURUSD_lgbm_T_20M.pkl → 20-minute timeframe
    
    EURUSD_lgbm_T_30M.pkl → 30-minute timeframe

  EURUSD_lgbm_bundle.pkl usually contains:
  
  The trained model
  
  Feature columns
  
  Metadata required for prediction

2️⃣ CSV_FILES
  This folder stores all historical market data and logs used by the system.
  
  Key files include:
  
  BACKTEST_DATA.csv → Cleaned dataset used for backtesting
  
  dukascopy_EURUSD_5M.csv → Dukascopy historical data
  
  MT5_*_Exchange_Rate_Dataset.csv → Data collected directly from MetaTrader 5
  
  Trade_log.csv → Stores executed trades and results
  
  These CSV files are used for:
  
  Training models
  
  Backtesting strategies
  
  Evaluating performance
  
  Debugging prediction behavior

3️⃣ PY_FILES

  This folder contains all Python scripts responsible for logic and execution.
  
  Important scripts:
  
  ALL_PROCESS.py
  Handles data preprocessing, indicator creation, and feature engineering.
  
  Preprocessing.py
  Cleans raw market data and prepares it for training or prediction.
  
  Load_model.py
  Loads trained models from the ALL_MODELS directory.
  
  ALL_PRED_NXT.py
  Runs predictions for the next candle across multiple timeframes.
  
  PRED_NEXT.py
  Single prediction execution for real-time or testing purposes.
  
  ALL_BACKTEST.py
  Backtests trading strategies using historical data and logs results.
  
  Get_dataMT5.py
  Collects live or historical data from MetaTrader 5.
  
  Get_dataYF.py
  Collects market data using Yahoo Finance.
  
  Dukascopy_Data.py
  Downloads and processes Dukascopy Forex data.
  
  func.py
  Contains helper and utility functions used across scripts.

4️⃣ requirements.txt

  This file lists all Python dependencies required to run the project.
  numpy
  pandas
  scikit-learn
  lightgbm
  catboost
  joblib
  MetaTrader5
  ta


Installation Guide
Step 1: Clone the Repository

Plain Text

git clone https://github.com/yourusername/forex-trading.git
cd FOREX-TRADING

Step 2: Install Dependencies

Plain Text

pip install -r requirements.txt

How to Use the Project
🔹 Run Data Processing

Plain Text

python PY_FILES/ALL_PROCESS.py


This prepares raw CSV data for training or prediction.

🔹 Run Backtesting

Plain Text

python PY_FILES/ALL_BACKTEST.py


This will:
Simulate trades on historical data
Calculate win rate
Log trades into Trade_log.csv

🔹 Predict Next Candle Direction
python PY_FILES/ALL_PRED_NXT.py
or
python PY_FILES/PRED_NEXT.py


The model will:

Load the appropriate timeframe model

Predict UP or DOWN

Output probability scores

YouTube Tutorials & Explanations

I explain the logic, math, and implementation behind this project on my YouTube channel:

📺 Ezee Kits YouTube Channel
🔗 https://www.youtube.com/@Ezee_Kits

The channel covers:

Python programming

Machine learning

Forex trading automation

Data analysis

Engineering & DIY concepts


Disclaimer ⚠️
  This project is for educational and research purposes only.
  Forex trading carries a high level of risk, and you should never trade with money you cannot afford to lose.
  
  The author is not responsible for any financial losses.

#Author
Ezee Kits
Electrical & Electronics Engineer
Python Developer | Machine Learning Enthusiast
YouTube: https://www.youtube.com/@Ezee_Kits
