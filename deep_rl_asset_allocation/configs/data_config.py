"""Config for the 30 Stocks from the Dow Jones"""

import datetime

# total number of stocks in our portfolio
STOCK_DIM = 30

# training is from: 2009 Jan 1st - 2015 Nov 30th
TRAINING_START = datetime.date(2009, 1, 1)
TRAINING_END = datetime.date(2015, 9, 30)
# validation is from: 2015 Oct 1st - 2015 Dec 31st
VALIDATION_START = datetime.date(2015, 10, 1)
VALIDATION_END = datetime.date(2015, 12, 31)
# testing is from: 2016 Jan 1st - 2020 May 8th
TESTING_START = datetime.date(2016, 1, 4)
TESTING_END = datetime.date(2020, 5, 8)

NUMBER_OF_TRADING_DAYS_PER_YEAR = 252
# rebalance_window is the number of months to retrain the model
REBALANCE_WINDOW = 63
# validation_window is the number of months to validation the model and select for trading
VALIDATION_WINDOW = 63
