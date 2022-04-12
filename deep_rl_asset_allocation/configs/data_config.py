"""Config for the 30 Stocks from the Dow Jones"""

import datetime

# Default Cols in our dataset
COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
#
TRAINING_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker', 'MACD', 'RSI', 'CCI', 'ADX', 'Turbulence']

# total number of stocks in our portfolio
TICKERS = [
    "^DJI",
    # Apple:
    'AAPL',
    # Amex
    'AXP',
    # British Airways
    'BA',
    # Caterpillar
    'CAT',
    # Cisco
    'CSCO',
    # Chevron
    'CVX',
    # DuPont: https://finance.yahoo.com/quote/DD/
    'DD',
    # Walt Disney
    'DIS',
    # Goldman Sachs
    'GS',
    # Home Depot
    'HD',
    # IBM
    'IBM',
    # Intel
    'INTC',
    # Johnson & Johnson
    'JNJ',
    # JPMorgan Chase
    'JPM',
    # Coca-Cola Company
    'KO',
    # McDonald's
    'MCD',
    # 3M
    'MMM',
    # Merck
    'MRK',
    # Microsoft
    'MSFT',
    # Nike
    'NKE',
    # Pfizer
    'PFE',
    # Procter & Gamble
    'PG',
    # Raytheon Technologies
    'RTX',
    # Travelers
    'TRV',
    # UnitedHealth
    'UNH',
    # Visa
    'V',
    # Verizon
    'VZ',
    # Walgreens
    'WBA',
    # Walmart
    'WMT',
    # ExxonMobil
    'XOM',
]
STOCK_DIM = len(TICKERS) - 1  # we include ^DJI

# training is from: 2009 Jan 1st - 2015 Nov 30th
TRAINING_START = datetime.date(2009, 1, 1)
TRAINING_END = datetime.date(2015, 9, 30)
# validation is from: 2015 Oct 1st - 2015 Dec 31st
VALIDATION_START = datetime.date(2015, 10, 1)
VALIDATION_END = datetime.date(2015, 12, 31)
# testing is from: 2016 Jan 1st - 2020 May 8th
TESTING_START = datetime.date(2016, 1, 4)
TESTING_END = datetime.date(2022, 1, 1)  # datetime.date(2020, 5, 8) or datetime.date(2022, 1, 1)

NUMBER_OF_TRADING_DAYS_PER_YEAR = 252
# rebalance_window is the number of months to retrain the model
REBALANCE_WINDOW = 63
# validation_window is the number of months to validation the model and select for trading
VALIDATION_WINDOW = 63
