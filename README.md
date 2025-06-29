📈 Smart Stock Tracking and Signal System
This project is a web-based application designed to track specified stock assets using near real-time data, generate buy/sell signals based on advanced technical analysis indicators, and manage open positions. It provides a foundation for identifying potential short-to-medium term opportunities in financial markets and automating portfolio management.

✨ Features
Near Real-Time Data Fetching: Retrieves stock prices and volume data on an hourly and daily basis using the yfinance library.

Comprehensive Technical Analysis:

Momentum: RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence).

Trend: SMA (Simple Moving Averages - 20, 50, 200), PSAR (Parabolic SAR).

Volatility: Bollinger Bands.

Volume: OBV (On Balance Volume).

Candlestick Patterns: Recognizes Bullish Engulfing and Bearish Engulfing patterns.

Intelligent Signal Generation:

Buy Signal: Triggered by the simultaneous fulfillment of multiple criteria, including high RSI, SMA crossover, strong MACD, high volume, bullish engulfing pattern, confirmation from a long-term upward trend, and conceptual machine learning model prediction.

Sell Signal: Triggered by any of the exit conditions such as price falling below PSAR, closing below SMA20, bearish MACD signal, or bearish engulfing pattern.

Long-Term Trend Analysis: Determines the general long-term trend of a stock (Uptrend, Downtrend, Sideways) using daily data (e.g., SMA50 above SMA200 and positive daily MACD), and uses this information to filter buy signals.

Position Management: Tracks open positions (entry price and time), automatically logs entry/exit actions with signals, and calculates profit/loss upon position closure.

Web-Based User Interface: Built with a Flask backend and HTML/CSS/JavaScript (Tailwind CSS) frontend, offering a clean and user-friendly dashboard displaying live stock statuses, signals, and open positions.

Detailed Logging: All operations, signals, and errors are recorded in log files.

🛠️ Technologies Used
Backend: Python (Flask)

Data Fetching: yfinance

Technical Analysis: ta (Technical Analysis) library

Data Manipulation: pandas, numpy

Frontend: HTML, CSS (Tailwind CSS), JavaScript

Data Storage: CSV (Simple file-based storage for open positions)

Machine Learning (Conceptual): joblib (For model loading; currently integrates a dummy model for demonstration purposes.)

🔮 Future Enhancements
This project has continuous development potential. Possible future enhancements include:

Integration with real-time data providers (e.g., Alpaca, Polygon.io).

Broker API integration for automated buy/sell order execution.

More sophisticated and trained Machine Learning models (e.g., LSTM, Transformers).

Comprehensive backtesting and strategy optimization module.

Signal notifications via email, Telegram, or mobile app.

User profiles and multi-strategy support.

⚠️ Legal Disclaimer
This project is for educational and demonstration purposes only. The signals or analyses generated here should not be construed as financial advice. Trading in financial markets involves significant risks and may result in loss of capital. Do not make investment decisions without conducting your own research or consulting with a licensed financial professional.
