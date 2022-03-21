import unittest

import deep_rl_asset_allocation.configs.paths_config as cfg_paths
import deep_rl_asset_allocation.utils.data_loader_utils as data_loader_utils
import gym
import numpy as np
from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from deep_rl_asset_allocation.envs.wrappers.continuous_to_discrete_dict_action_wrapper import \
    ContinuousToDiscreteDictActionWrapper
from gym.wrappers import FlattenObservation
from stable_baselines3.common import env_checker

GYM_ID = 'stock-env-v0'


class TestMultiStockEnv(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMultiStockEnv, self).__init__(*args, **kwargs)
        # load DIJA data
        djia_data_dict = data_loader_utils.get_train_val_test_djia_data()
        self.df_train = djia_data_dict["df_train"]

    def _step_openai_gym(self, env):
        """test for basic functionality of an Open AI gym env"""

        action_space = env.action_space
        obs_space = env.observation_space

        for idx in range(10):

            obs = env.reset()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            print(f"\nIdx: {idx+1}")
            # print('Action:')
            # print(action)
            print('Observation:')
            print(observation[0])  # first value is the account balance
            # print('Reward:')
            # print(reward)
            # print(f'Done: {done}')
            # print(f'Info: {info}')

    def test_env(self):
        # init from class definition (not recommended by OpenAI gym)
        env_train = MultiStockEnv(self.df_train)
        # run simple tests
        self._step_openai_gym(env=env_train)

    def test_gym_make_env(self):
        # init with gym.make()
        env_train = gym.make(GYM_ID, df=self.df_train)
        # run simple tests
        self._step_openai_gym(env=env_train)

    def test_flatten_observation_wrapper_env(self):
        env_train = FlattenObservation(gym.make(GYM_ID, df=self.df_train))
        # run simple tests
        self._step_openai_gym(env=env_train)

        # TODO: Broken because of dict action. fix upstream in sb3
        # Implicitly closes the environment
        # env_checker.check_env(env=env_train, warn=True, skip_render_check=True)

    def test_discrete_action_wrapper_env(self):
        env_train = ContinuousToDiscreteDictActionWrapper(gym.make(GYM_ID, df=self.df_train))
        # check compliance for stable baselines 3
        env_checker.check_env(env=env_train, warn=True, skip_render_check=True)
        # run simple tests
        self._step_openai_gym(env=env_train)


if __name__ == '__main__':
    # run all test.
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(TestMultiStockEnv("test_flatten_observation_wrapper_env"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
