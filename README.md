# AI Forex Trading System using LightGBM & MetaTrader 5

## Overview

This repository contains a complete AI-powered Forex Trading System developed from the ground up using Python, MetaTrader 5 (MT5), LightGBM, Optuna, SHAP, and advanced feature engineering techniques.

Unlike traditional rule-based trading bots, this system uses Machine Learning to identify high-probability trading opportunities by learning market behavior directly from historical price data.

The project was designed with a strong emphasis on:

* Institutional-grade feature engineering
* Time-series safe model training
* Robust forward testing
* Risk management
* Live trading compatibility
* Generalization across multiple currency pairs

This project represents several months of research, experimentation, optimization, debugging, and continuous improvement.

---

# Project Objectives

The primary objective of this project is to build an automated trading system capable of:

* Predicting high-probability market direction
* Filtering low-quality trading opportunities
* Managing risk automatically
* Working across multiple Forex pairs and commodities
* Remaining profitable under real market conditions

---

# Main Technologies

## Programming Language

* Python

## Machine Learning

* LightGBM
* Scikit-Learn
* Optuna
* SHAP

## Data Processing

* Pandas
* NumPy

## Trading Platform

* MetaTrader 5
* MT5 Python API

## Visualization

* Matplotlib

## Model Storage

* Joblib

---

# Trading Instruments

The system has been developed and tested on multiple financial instruments including:

* EURUSD
* GBPUSD
* AUDUSD
* AUDJPY
* CADJPY
* NZDCAD
* NZDUSD
* USDCHF
* USDCAD
* USDSEK
* BTCUSD
* XAUUSD (Gold)
* XAGUSD (Silver)

The architecture is designed so that additional instruments can easily be integrated.

---

# Dataset Generation

Historical market data is collected directly from MetaTrader 5.

Each dataset contains:

* Date
* Open
* High
* Low
* Close
* Tick Volume
* Spread

Datasets are automatically cleaned and prepared before feature engineering.

---

# Market Session Filtering

One major discovery during development was that certain trading hours consistently experienced extremely high spreads.

To improve live trading performance, the system automatically removes the following hours before feature engineering:

* 21:00
* 22:00

This prevents the model from learning trading opportunities that are practically impossible to execute profitably due to spread expansion.

---

# Feature Engineering

One of the strongest parts of this project is the feature engineering pipeline.

Hundreds of market features are generated automatically.

Examples include:

## Trend Features

* EMA Alignment
* EMA Distance
* EMA Slope
* Multi-Timeframe Trend
* Trend Persistence

## Volatility Features

* ATR
* ATR Expansion
* ATR Acceleration
* Volatility Regime
* Range Expansion
* Bollinger Band Compression
* Bollinger Band Expansion

## Momentum Features

* RSI
* RSI Velocity
* RSI Acceleration
* MACD
* MACD Slope
* Momentum Persistence

## Breakout Features

* Breakout Strength
* Breakdown Strength
* High Distance
* Low Distance
* Compression Detection

## Pair-Specific Institutional Features

Different financial instruments behave differently.

Custom institutional features were created specifically for instruments such as:

* Gold
* Silver
* Bitcoin
* EURUSD
* GBPJPY
* AUDUSD
* AUDJPY
* CADJPY

These features capture market characteristics unique to each instrument.

---

# Correlation Filtering

Before model training, highly correlated features are automatically removed.

This reduces redundancy while improving model stability.

Correlation filtering is performed only on the training portion of the dataset to avoid future data leakage.

---

# SHAP Feature Selection

After correlation filtering, SHAP (SHapley Additive exPlanations) is used for feature selection.

Instead of relying on feature importance from a single model, the system:

* Uses TimeSeriesSplit
* Trains multiple LightGBM models
* Computes SHAP values for every fold
* Calculates mean SHAP importance
* Measures feature stability across folds
* Selects only the most reliable features

This greatly improves model generalization.

---

# Hyperparameter Optimization

The project uses Optuna to optimize LightGBM parameters.

Parameters optimized include:

* Learning Rate
* Number of Leaves
* Maximum Depth
* Number of Estimators
* Minimum Child Samples
* Feature Sampling
* Row Sampling

The optimization process is entirely automated.

---

# Time Series Safe Training

Unlike traditional random train-test splits, this project uses:

TimeSeriesSplit

This prevents future information from leaking into the past and produces much more realistic model evaluation.

---

# Label Generation Engine

The project contains a custom labeling engine.

Labels are generated using:

* ATR-based Take Profit
* ATR-based Stop Loss
* Forward market simulation
* Ambiguous trade detection
* Noise buffer

Trades that hit both TP and SL within the prediction horizon are automatically discarded.

This produces much cleaner training labels.

---

# Risk Management

The trading system includes:

* ATR-based Stop Loss
* ATR-based Take Profit
* Spread Ratio Filtering
* Kelly Criterion Position Sizing
* Dynamic Risk Control

---

# Spread Handling

Rather than ignoring spread, the project explicitly models it.

The system calculates:

Spread Ratio = Spread / Stop Loss Distance

Only trades with acceptable spread conditions are considered.

This makes live trading significantly more realistic.

---

# Forward Testing

Every model is evaluated using:

* Historical Backtesting
* Forward Testing
* Multiple Market Conditions

Performance metrics include:

* Win Rate
* Profit Factor
* Maximum Drawdown
* Sharpe Ratio
* Expectancy
* Final Balance

---

# Machine Learning Workflow

1. Download MT5 historical data

2. Remove high-spread market hours

3. Generate hundreds of market features

4. Remove correlated features

5. Optimize LightGBM using Optuna

6. Perform SHAP-based feature selection

7. Retrain using selected features

8. Validate using TimeSeriesSplit

9. Save optimized model

10. Deploy for live trading

---

# Project Highlights

* Complete end-to-end trading pipeline
* Institutional feature engineering
* Automated feature selection
* Automatic hyperparameter tuning
* Time-series safe validation
* Live MT5 integration
* Multi-asset support
* Risk management
* Spread-aware execution
* Modular architecture

---

# Repository Structure

```text
├── Data/
├── Models/
├── Feature_Engineering/
├── Training/
├── Backtesting/
├── Live_Trading/
├── Utilities/
├── CSV_Files/
├── README.md
```

---

# Skills Demonstrated

* Python Programming
* Machine Learning
* Quantitative Finance
* Financial Data Analysis
* Feature Engineering
* Time Series Forecasting
* Hyperparameter Optimization
* Explainable AI (SHAP)
* Algorithmic Trading
* MetaTrader 5 Automation
* Data Cleaning
* Performance Optimization
* Software Engineering

---

# Future Improvements

Planned enhancements include:

* Deep Learning Models
* Transformer-based Time Series Models
* Reinforcement Learning
* Portfolio Optimization
* Multi-Timeframe Ensemble Models
* Online Learning
* Cloud Deployment
* Web Dashboard
* Real-Time Monitoring
* Strategy Comparison Framework

---

# Portfolio

My professional portfolio:

**https://ezee-kits-portfolio.onrender.com/**

---

# YouTube

I regularly publish tutorials and technical content covering:

* Python
* Machine Learning
* Artificial Intelligence
* Algorithmic Trading
* Data Science
* Electrical Engineering
* Software Development

YouTube Channel:

**https://www.youtube.com/@EzeeKits**

---

# Contact

I am available for freelance projects involving:

* Python Development
* Machine Learning
* AI Systems
* Trading Bots
* Data Analysis
* Automation
* Electrical Engineering Software
* Custom Algorithm Development

Portfolio:
https://ezee-kits-portfolio.onrender.com/

GitHub:
https://github.com/Ezee-Kits/

YouTube:
https://www.youtube.com/@EzeeKits

---

# License

This repository is intended for educational, research, and portfolio purposes.

Please note that trading financial markets involves significant risk. Past performance does not guarantee future results. This project should not be considered financial advice.
