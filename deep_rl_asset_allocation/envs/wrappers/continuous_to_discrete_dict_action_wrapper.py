from tracemalloc import start

import gym
import numpy as np
from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from gym import spaces


class ContinuousToDiscreteDictActionWrapper(gym.ActionWrapper):

    STOCK_DIM = 30

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, MultiStockEnv), 'Wrapped Env must be of type multi_stock_env'

        # Action Space is discrete with np.int from STRONG_SELL[-1], DO_NOTHING[0], STRONG_BUY[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(ContinuousToDiscreteDictActionWrapper.STOCK_DIM, ), dtype=np.int)

    def action(self, discrete_action):
        return discrete_action
