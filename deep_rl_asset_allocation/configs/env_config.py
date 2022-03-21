"""Configs for Trading Env"""

GYM_ID = 'stock-env-v0'

# see test_gym_env.ipynb
N_OBS = 181
N_ACTIONS = 30
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# shares normalization factor of 100 shares per trade
HMAX_NORMALIZE = 100
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4
# turbulence index: 90-150 reasonable threshold
TURBULENCE_THRESHOLD = 140
