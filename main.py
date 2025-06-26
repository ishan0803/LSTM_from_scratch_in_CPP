import yfinance as yf
import pandas as pd
import subprocess

# Take input from user
ticker = input("Enter stock ticker (e.g., ICICIBANK.NS): ").strip()
duration = input("Enter duration (e.g., 5y, 1y, 6mo): ").strip()

# Download data
data = yf.download(ticker, period=duration, interval='1d')

# Check if data is valid
if data.empty:
    print("Failed to fetch data. Please check the ticker and duration.")
    exit(1)

# Keep only 'Open' column and reset index
open_prices = data[['Open']]
open_prices.reset_index(inplace=True)

# Generate a CSV file
csv_filename = f"{ticker.replace('.', '_')}_open_prices_{duration}.csv"
open_prices.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' created successfully.")

# Optional: Run C++ predictor if compiled binary exists
run_predictor = input("Run predictor with this CSV? (y/n): ").strip().lower()
if run_predictor == 'y':
    try:
        subprocess.run(["./Lstm_predictor.exe", csv_filename])
    except FileNotFoundError:
        print("Predictor binary not found. Make sure it's compiled.")
