import os

import deep_rl_asset_allocation.configs.paths_config as cfg_paths
import deep_rl_asset_allocation.utils.data_loader_utils as data_loader_utils
import gym
import numpy as np
import pandas as pd
from deep_rl_asset_allocation.envs.multi_stock_env import MultiStockEnv
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.ppo import MlpPolicy
from torch.utils.tensorboard import SummaryWriter

GYM_ID = 'stock-env-v0'

PPO_TIMESTEPS = 100000
PPO_SAVE_FREQ = PPO_TIMESTEPS // 5

EXP_NUM = "ppo-1"
TRAINED_MODELS_BASE_DIR = cfg_paths.results_trained_models_dir + "/"
PPO_MONITOR_FILENAME = os.path.join(TRAINED_MODELS_BASE_DIR, f'{EXP_NUM}-montior-log')
PPO_CALLBACK_SAVED_MODEL_DIR = os.path.join(TRAINED_MODELS_BASE_DIR, f'{EXP_NUM}-callback-checkpoint-models/')
PPO_CALLBACK_NAME_PREFIX = os.path.join(PPO_CALLBACK_SAVED_MODEL_DIR, f'{EXP_NUM}')
PPO_TENSORBOARD_DIR = os.path.join(TRAINED_MODELS_BASE_DIR, f'{EXP_NUM}-callback-tensorboard/')
PPO_SAVED_MODEL_FILENAME = os.path.join(TRAINED_MODELS_BASE_DIR, f'{EXP_NUM}-model')


class TensorboardCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""

    def __init__(self, monitor_log_dir, save_best_model_path, tensorboard_log_dir, check_freq=1000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.monitor_log_dir = monitor_log_dir
        self.check_freq = check_freq
        self.verbose = verbose

        self.writer = SummaryWriter(tensorboard_log_dir, comment='PPO_GRADS_and_WEIGHTS')

        self.save_best_model_path = os.path.join(save_best_model_path, 'best_mean_reward_model')

        self.best_mean_reward = -np.inf
        self.history_for_best_mean_reward = 10

        self.policy = None

    def _on_step(self) -> bool:
        """
        See tutorial here https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
        """

        if self.policy == None:
            self.policy = self.model.policy.mlp_extractor.policy_net

        if self.n_calls % self.check_freq == 0:
            for tag, value in self.policy.named_parameters():
                tag = tag.replace('.', '/')
                if value.grad is None:
                    # print('No Grad Layer: ', tag.split('/'))
                    self.writer.add_histogram('weights/ppo/policy/' + tag, value.data.cpu().numpy(), self.n_calls)
                    pass
                else:
                    # print('Layer: ', tag.split('/'))
                    self.writer.add_histogram('weights/ppo/policy/' + tag, value.data.cpu().numpy(), self.n_calls)
                    self.writer.add_histogram('grads/ppo/policy/' + tag, value.grad.data.cpu().numpy(), self.n_calls)

            monitor_csv_dataframe = load_results(self.monitor_log_dir)
            # dataframe is loaded as: [index], [r], [l], [t]
            index = monitor_csv_dataframe['index'].to_numpy()
            rewards = monitor_csv_dataframe['r'].to_numpy()
            episode_lengths = monitor_csv_dataframe['l'].to_numpy()

            # TODO: log sharpe ratio

            if len(rewards) > self.history_for_best_mean_reward:

                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(rewards[-self.history_for_best_mean_reward:]) / np.mean(episode_lengths[-self.history_for_best_mean_reward:])
                self.writer.add_scalar("reward/mean_reward", mean_reward, global_step=len(rewards))

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.writer.add_scalar("reward/best_mean_reward", self.best_mean_reward, global_step=len(rewards))
                    # saving best model
                    self.model.save(self.save_best_model_path)

                    if self.verbose > 0:
                        print("Best mean reward: {:.2f}\nLast mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

        return True


def main():
    df_train, df_val = data_loader_utils.get_train_val_test_djia_data()
    env_train = Monitor(env=FlattenObservation(gym.make(GYM_ID, df=df_train, day=0)), filename=TRAINED_MODELS_BASE_DIR)

    checkpoint_callback = CheckpointCallback(
        save_freq=PPO_SAVE_FREQ,
        save_path=PPO_CALLBACK_SAVED_MODEL_DIR,
        name_prefix=PPO_CALLBACK_NAME_PREFIX,
    )

    custom_callback = TensorboardCallback(monitor_log_dir=TRAINED_MODELS_BASE_DIR,
                                          save_best_model_path=TRAINED_MODELS_BASE_DIR,
                                          tensorboard_log_dir=PPO_TENSORBOARD_DIR,
                                          check_freq=1000,
                                          verbose=1)

    ppo_callbacks = CallbackList([checkpoint_callback, custom_callback])

    model = PPO(MlpPolicy, env=env_train)
    model.learn(total_timesteps=PPO_TIMESTEPS, callback=custom_callback)
    model.save(path=PPO_SAVED_MODEL_FILENAME)


def get_daily_return(df):
    df['daily_return'] = df.account_value.pct_change(1)
    #df=df.dropna()
    sharpe_ratio = (252**0.5) * df['daily_return'].mean() / df['daily_return'].std()
    print(f'Sharpe Ratio: {sharpe_ratio*100:.2f} %')
    return df


def backtest_strat(df):
    strategy_ret = df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts


def get_account_value():
    df_account_value = pd.DataFrame()
    # for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    for training_iteration, idx in enumerate(range(rebalance_window + validation_window, len(unique_trade_dates), rebalance_window)):
        asset_memory_csv_filename = os.path.join(paths_config.results_csv_dir, f'account_value_test_{training_iteration}.csv')
        df = pd.read_csv(asset_memory_csv_filename)
        df_account_value = df_account_value.append(df, ignore_index=True)
    df_account_value = pd.DataFrame({'account_value': df_account_value['0']})
    sharpe_ratio = (252**0.5) * df_account_value.account_value.pct_change(1).mean() / df_account_value.account_value.pct_change(1).std()
    print(f'Sharpe Ratio: {sharpe_ratio*100:.2f} %')
    df_account_value = df_account_value.join(df_trade_date[63:].reset_index(drop=True))
    return df_account_value


if __name__ == '__main__':
    main()
