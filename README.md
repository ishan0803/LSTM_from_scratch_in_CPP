# Stock Price Prediction using LSTM Neural Network

## Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network for stock price prediction. It combines Python-based data fetching with a C++ implementation of the LSTM model, creating a hybrid system that leverages the strengths of both languages.

## Educational Value

This project serves as an excellent educational resource for understanding:

1. **Deep Learning Concepts**
   - Implementation of LSTM (Long Short-Term Memory) networks from scratch
   - Understanding of gates (forget, input, cell, output) in LSTM
   - Gradient descent and backpropagation through time
   - Hyperparameter tuning

2. **C++ Programming**
   - Object-oriented programming with structs and classes
   - Memory management and vector operations
   - Implementation of mathematical functions
   - Efficient numerical computations

3. **Python Integration**
   - Data fetching using yfinance API
   - Data preprocessing and CSV handling
   - Inter-process communication between Python and C++

## Architecture

### Python Component (`main.py`)
- Handles data acquisition using the `yfinance` library
- Allows user input for stock ticker and duration
- Preprocesses and exports data to CSV
- Interfaces with the C++ predictor

### C++ Component (`AI_CP_Final.cpp`)
- Implements the core LSTM neural network
- Features:
  - Input size: 1 (stock price)
  - Hidden layer size: 32 neurons
  - Sequence length: 10 time steps
  - Customizable learning rate and gradient clipping

#### LSTM Implementation Details
- Custom implementation of activation functions (sigmoid, tanh)
- Complete LSTM cell structure with:
  - Weight matrices (Wf, Wi, Wc, Wo)
  - Bias vectors (bf, bi, bc, bo)
  - State tracking for backpropagation
- Gradient clipping for training stability

## Key Features

1. **Real-time Data Fetching**
   - Direct stock data acquisition from Yahoo Finance
   - Support for various time periods (e.g., 5y, 1y, 6mo)
   - Flexible ticker symbol input

2. **Advanced LSTM Architecture**
   - Forward and backward propagation
   - State management for sequence learning
   - Gradient clipping for stable training

3. **Hybrid Language Approach**
   - Python for data handling
   - C++ for computational efficiency
   - Seamless integration between components

## Technical Specifications

- **Hyperparameters**
  - Input Size: 1 (stock price)
  - Hidden Layer Size: 32
  - Output Size: 1
  - Sequence Length: 10
  - Learning Rate: 0.001
  - Gradient Clip: 5.0

## Usage

1. Run the Python script:
   ```bash
   python main.py
   ```

2. Enter the requested information:
   - Stock ticker (e.g., ICICIBANK.NS)
   - Duration (e.g., 5y, 1y, 6mo)

3. The system will:
   - Download the stock data
   - Create a CSV file
   - Optionally run the C++ predictor

## Dependencies

### Python
- yfinance
- pandas

### C++
- Standard Template Library (STL)
- C++11 or higher

## Educational Applications

This project is particularly valuable for:
- Advanced programming courses
- Machine learning education
- Financial computing classes
- Algorithm implementation studies

The combination of Python and C++ also makes it an excellent example of:
- Language interoperability
- Performance optimization
- Real-world application design

## Implementation Notes

- Fixed random seed (42) for reproducibility
- Gradient clipping for training stability
- Modular design for easy modification and experimentation
- Comprehensive error handling and data validation

This project demonstrates the practical implementation of complex machine learning concepts while maintaining educational clarity and code readability.
