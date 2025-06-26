# Stock Price Prediction with Custom LSTM (C++/Python)

## Project Overview
This project implements a robust, end-to-end stock price prediction system using a custom Long Short-Term Memory (LSTM) neural network in C++ and a Python-based data pipeline. It demonstrates advanced skills in machine learning, quantitative analysis, and high-performance software engineering, applied to real-world financial data.

## Technical Stack
- **Languages:** C++ (core ML model), Python (data pipeline)
- **Libraries:** STL, yfinance, pandas
- **ML Concepts:** LSTM cell architecture, sequence modeling, time series forecasting, gradient clipping, hyperparameter tuning

## Architecture
- **Python (`main.py`):**
  - Fetches historical stock data using yfinance
  - Preprocesses and exports data to CSV
  - Optionally triggers the C++ predictor
- **C++ (`Lstm_predictor.cpp`):**
  - Implements LSTM neural network from scratch
  - Handles training and prediction
  - Outputs results for analysis

## Key Features & Skills Demonstrated
- Custom LSTM implementation (no external ML libraries)
- Automated, reproducible data pipeline
- Cross-language integration (Python ↔ C++)
- Efficient memory and state management
- Real-world time series forecasting
- Quantitative evaluation and reporting

## Results: Nifty 50 R² Scores
| Period | R² Score |
|--------|----------|
| 5y     | 0.7769   |
| 1y     | 0.5370   |
| 6m     | 0.2870   |

> These results reflect the model's ability to capture both long-term and short-term trends in financial data.

## How to Use
1. Run the Python script:
   ```bash
   python main.py
   ```
2. Enter the stock ticker and duration (e.g., 5y, 1y, 6mo)
3. The script downloads data, creates a CSV, and can run the C++ predictor if desired

## Dependencies
- **Python:** yfinance, pandas
- **C++:** STL, C++11 or higher

---
For more details or collaboration, see the code or contact via GitHub.
