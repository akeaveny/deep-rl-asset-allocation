import os
import pprint
import time
from xml.dom import INDEX_SIZE_ERR

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deep_rl_asset_allocation.configs import (data_config, env_config, paths_config)
from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from deep_rl_asset_allocation.utils import data_loader_utils
from gym.wrappers import FlattenObservation
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.vec_env import DummyVecEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

TIMESTEPS = 25000


def train_A2C(env_train, model_name, timesteps=TIMESTEPS):
    """A2C model with Stable-Baseline3"""
    a2c_saved_model_filename = os.path.join(paths_config.results_trained_models_dir, f'{model_name}_{timesteps}')
    # print(f"Saved Model to {a2c_saved_model_filename}")

    start = time.time()
    # model = A2C(MlpPolicy, env_train, ent_coef=0.005)  # nminibatches = 8
    model = A2C(MlpPolicy, env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(a2c_saved_model_filename)
    training_time = (time.time() - start) / 60
    print(f'Training time (A2C): {training_time:.1f} minutes')
    return model


def validate_model(model, df_val, env_val) -> None:
    obs = env_val.reset()
    for idx in range(len(df_val.index.unique())):
        action, state = model.predict(obs)
        obs, rewards, done, info = env_val.step(action)


def test_model(model, df_test, env_test, training_iteration):
    obs = env_test.reset()
    for idx in range(len(df_test.index.unique())):
        action, state = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        if idx == (len(df_test.index.unique()) - 2):
            last_obs = env_test.render()

    # getting values from obs dict
    account_balance = last_obs["account_balance"][0]
    adjusted_close_price = last_obs["adjusted_close_price"]
    shares_owned_fractional = last_obs["shares_owned_fractional"]
    moving_average_convergence_divergence = last_obs["moving_average_convergence_divergence"]
    relative_strength_index = last_obs["relative_strength_index"]
    commodity_channel_index = last_obs["commodity_channel_index"]
    average_directional_index = last_obs["average_directional_index"]
    total_assets_value = np.array(account_balance) + np.sum(np.array(adjusted_close_price) * np.array(shares_owned_fractional))
    print(f"Final Observation: \tAccount Balance: ${account_balance:.0f} \tTotal Asset Value: ${total_assets_value:.0f}")

    df_last_state = pd.DataFrame({
        'account_balance': int(account_balance),
        'total_assets_value': int(total_assets_value),
        'shares_owned_fractional': shares_owned_fractional,
        'adjusted_close_price': adjusted_close_price,
        'moving_average_convergence_divergence': moving_average_convergence_divergence,
        'relative_strength_index': relative_strength_index,
        'commodity_channel_index': commodity_channel_index,
        'average_directional_index': average_directional_index
    })

    final_observation_csv_filename = os.path.join(paths_config.results_csv_dir, f'final_observation_test_{training_iteration}.csv')
    df_last_state.to_csv(final_observation_csv_filename, index=False)
    return last_obs


def main():
    df_djia = data_loader_utils.load_preprocessed_djia_data()
    djia_data_dict = data_loader_utils.get_train_val_test_djia_data()
    df_train, df_val, df_test = djia_data_dict["df_train"], djia_data_dict["df_val"], djia_data_dict["df_test"]

    # get unique trade dates
    unique_trade_dates = df_djia[(df_djia.date > str(data_config.VALIDATION_START)) & (df_djia.date <= str(data_config.TESTING_END))].date.unique()

    # get turbulence threshold
    insample_turbulence = df_train
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['date'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

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
        print(f'val-start: {idx1}, val-end: {idx2} and test-end: {idx3} -> total: {len(unique_trade_dates)} months')

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
        end_index = df_djia.index[df_djia["date"] == val_start_date].to_list()[-1]
        start_index = end_index - (validation_window * 30 + 1)  # 30 days per month
        historical_turbulence = df_djia.iloc[start_index:(end_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['date'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)
        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print(f"Turbulence Threshold: {turbulence_threshold:.2f}")

        # TODO: init envs with DummyVecEnv([lambda: ])
        env_train = FlattenObservation(
            gym.make(env_config.GYM_ID, df=_df_train, split="train", training_iteration=training_iteration, turbulence_threshold=turbulence_threshold))
        env_val = FlattenObservation(
            gym.make(env_config.GYM_ID, df=_df_val, split="val", training_iteration=training_iteration, turbulence_threshold=turbulence_threshold))
        env_test = FlattenObservation(
            gym.make(env_config.GYM_ID,
                     df=_df_test,
                     split="test",
                     previous_observation=previous_observation,
                     training_iteration=training_iteration,
                     turbulence_threshold=turbulence_threshold))

        print(f"Training A2C ...")
        model_a2c = train_A2C(env_train, model_name=f"a2c_djia_training_iteration_{training_iteration}_timesteps", timesteps=TIMESTEPS)
        print(f"Validating A2C ...")
        validate_model(model=model_a2c, df_val=_df_val, env_val=env_val)
        print(f"Testing A2C ...")
        previous_observation = test_model(model=model_a2c, df_test=_df_test, env_test=env_test, training_iteration=training_iteration)


if __name__ == '__main__':
    main()
