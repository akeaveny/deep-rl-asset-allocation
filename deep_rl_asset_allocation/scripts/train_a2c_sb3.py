import os
import pprint
import shutil
import time
from xml.dom import INDEX_SIZE_ERR

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_rl_asset_allocation.configs import (data_config, env_config, paths_config)
from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from deep_rl_asset_allocation.utils import data_loader_utils, experiment_utils
from gym.wrappers import FlattenObservation
from stable_baselines3 import A2C, DDPG, PPO, TD3
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.vec_env import DummyVecEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setting random seed for reproducibility
experiment_utils.set_random_seed(seed=env_config.SEED)
# Removing old results
experiment_utils.init_directories_for_training()


def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model with Stable-Baseline3"""
    a2c_saved_model_filename = os.path.join(paths_config.results_trained_models_dir, f'{model_name}')

    start = time.time()
    # """A2C model with Stable-Baseline3"""
    model = A2C(MlpPolicy, env_train, verbose=0, seed=env_config.SEED, tensorboard_log=paths_config.results_tensorboard_dir)
    # model = PPO(MlpPolicy, env_train, ent_coef=0.005, verbose=0, seed=env_config.SEED, tensorboard_log=paths_config.results_tensorboard_dir)
    model.learn(total_timesteps=timesteps)
    model.save(a2c_saved_model_filename)
    print(f"Saved trained A2C model to: {a2c_saved_model_filename}")
    training_time = (time.time() - start) / 60
    print(f'Training time (SB3-A2C): {training_time:.1f} minutes')

    return model


def main():
    df_djia = data_loader_utils.load_preprocessed_djia_data()
    djia_data_dict = data_loader_utils.get_train_val_test_djia_data()
    df_train, df_val, df_test = djia_data_dict["df_train"], djia_data_dict["df_val"], djia_data_dict["df_test"]

    # get unique trade dates
    training_dates = df_djia[(df_djia.Date > data_config.TRAINING_START) & (df_djia.Date <= data_config.VALIDATION_START)].Date.unique()
    len_training_dates = len(training_dates)
    unique_trade_dates = df_djia[(df_djia.Date > data_config.VALIDATION_START) & (df_djia.Date <= data_config.TESTING_END)].Date.unique()

    # get turbulence threshold
    insample_turbulence = df_train
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['Date'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.Turbulence.values, .90)

    # rebalance_window is the number of months to retrain the model
    rebalance_window = data_config.REBALANCE_WINDOW
    # validation_window is the number of months to validation the model and select for trading
    validation_window = data_config.VALIDATION_WINDOW

    # init for test env
    previous_observation = {}
    for training_iteration, idx in enumerate(range((rebalance_window + validation_window), len(unique_trade_dates), rebalance_window)):
        # if training_iteration == 1:
        # return
        print(f'\nTraining Iteration: {training_iteration+1}')

        # initial state is empty
        initial = True if (idx - rebalance_window - validation_window == 0) else False

        # get start and end dates indexes
        idx1 = idx - rebalance_window - validation_window
        idx2 = idx - rebalance_window
        idx3 = idx
        print(f'Total Months: {len(unique_trade_dates)}, val-start: {idx1}, val-end: {idx2}, test-end: {idx3}')

        val_start_date = unique_trade_dates[idx1]
        val_end_date = unique_trade_dates[idx2]
        test_end_date = unique_trade_dates[idx3]
        print(f'Training:\t Start Date: {data_config.TRAINING_START}\t\tEnd Date: {val_start_date}')
        print(f'Validation:\t Start Date: {val_start_date}\t\tEnd Date: {val_end_date}')
        print(f'Testing:\t Start Date: {val_end_date}\t\tEnd Date: {test_end_date}')

        _df_train = data_loader_utils.get_data_between_dates(df_djia, start=data_config.TRAINING_START, end=val_start_date)
        _df_val = data_loader_utils.get_data_between_dates(df_djia, start=val_start_date, end=val_end_date)
        _df_test = data_loader_utils.get_data_between_dates(df_djia, start=val_end_date, end=test_end_date)

        # get mean of historical turbulence
        end_index = df_djia.index[df_djia["Date"] == val_start_date].to_list()[-1]
        start_index = end_index - (validation_window * 30 + 1)  # 30 days per month
        historical_turbulence = df_djia.iloc[start_index:(end_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['Date'])
        historical_turbulence_mean = np.mean(historical_turbulence.Turbulence.values)
        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.Turbulence.values, 1)
        turbulence_threshold = 1000
        print(f"Turbulence Threshold: {turbulence_threshold:.2f}")

        # TODO: init envs with DummyVecEnv([lambda: ])
        env_train = DummyVecEnv([
            lambda: FlattenObservation(
                gym.make(env_config.GYM_ID, df=_df_train, split="train", training_iteration=training_iteration, turbulence_threshold=turbulence_threshold))
        ])

        print(f"Training A2C ...")
        model_a2c = train_A2C(env_train, model_name=f"sb3_a2c_training_iteration_{training_iteration}")


if __name__ == '__main__':
    main()
