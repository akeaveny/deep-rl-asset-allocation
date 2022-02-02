from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from gym.envs.registration import register

register(
    id='stock-env-v0',
    entry_point='deep_rl_asset_allocation.envs:MultiStockEnv',
)
