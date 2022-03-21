import warnings
from datetime import datetime

import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio as pf
from dateutil.relativedelta import relativedelta
from yahoofinancials import YahooFinancials

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 200
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_performance_summary(returns):
    '''
    Calculate selected performance evaluation metrics using provided returns.
    
    Parameters
    ------------
    returns : pd.Series
        Series of returns we want to evaluate

    Returns
    -----------
    stats : pd.Series
        The calculated performance metrics
        
    '''
    stats = {
        'annualized_returns': ep.annual_return(returns),
        'cumulative_returns': ep.cum_returns_final(returns),
        'annual_volatility': ep.annual_volatility(returns),
        'sharpe_ratio': ep.sharpe_ratio(returns),
        'sortino_ratio': ep.sortino_ratio(returns),
        'max_drawdown': ep.max_drawdown(returns)
    }
    return pd.Series(stats)


def visualize_results(df, title, currency='$'):
    '''
    Visualize the overview of the trading strategy including:
    * the evolution of the capital
    * price of the asset, together with buy/sell signals
    * daily returns
    
    Parameters
    ------------
    df : pd.DataFrame
        Performance DataFrame obtained from `zipline`
    title : str
        The title of the plot
    currency : str
        The symbol of the considered currency
        
    '''
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=[16, 9])

    # portfolio value
    df.portfolio_value.plot(ax=ax[0])
    ax[0].set_ylabel('portfolio value in ' + currency)

    # daily returns
    df.returns.plot(ax=ax[1])
    ax[1].set_ylabel('daily returns')

    fig.suptitle(title, fontsize=16)
    plt.legend()
    plt.show()

    print('Final portfolio value (including cash): {amount}{currency}'.format(amount=np.round(df.portfolio_value[-1], 2), currency=currency))
