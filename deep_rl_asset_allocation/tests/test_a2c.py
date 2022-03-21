import warnings

import gym
from deep_rl_asset_allocation.agents import a2c
from deep_rl_asset_allocation.configs import (data_config, env_config,
                                              paths_config)
from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from deep_rl_asset_allocation.utils import data_loader_utils
from gym.wrappers import FlattenObservation

warnings.filterwarnings("ignore")


def test_a2c_select_action(env, model, _STEPS=10):
    observation = env.reset()
    for idx in range(_STEPS):
        # select action from policy
        action = model._select_action(observation)
        print(f"action: {action}")
        # take the action
        observation, reward, done, info = env.step(action)


def test_a2c_training(env, model, num_steps=500, total_timesteps=25000):
    model.learn(num_steps=num_steps, total_timesteps=total_timesteps)


if __name__ == "__main__":
    df_djia = data_loader_utils.load_preprocessed_djia_data()
    djia_data_dict = data_loader_utils.get_train_val_test_djia_data()
    df_train, df_val, df_test = djia_data_dict["df_train"], djia_data_dict["df_val"], djia_data_dict["df_test"]

    env_train = FlattenObservation(gym.make(env_config.GYM_ID, df=df_train, split="train"))
    n_actions = env_config.N_ACTIONS
    n_obs = env_config.N_OBS

    # create Actor Critic Policy
    policy = a2c.ContinousPolicy(n_obs, n_actions)
    # init Actor Critic
    model = a2c.A2C(env_train, policy)

    # test_a2c_select_action(env_train, model)
    test_a2c_training(env_train, model)
