import os
import pprint
import time

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deep_rl_asset_allocation.configs import (data_config, env_config,
                                              paths_config)
from gym import spaces
from gym.utils import seeding

# "Agg" backend is for writing to file, not for rendering in a window
# matplotlib.use('Agg')


class MultiStockEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df,
        day=0,
        split="train",
        training_iteration=0,
        previous_observation=[],
        turbulence_threshold=env_config.TURBULENCE_THRESHOLD,
    ):

        # info about the trajectory
        self.env_info = {
            "split": split,
            "training_iteration": training_iteration,
            "previous_observation": previous_observation,
            "day": day,
            "max_days": (len(df.index.unique()) - 1),
            "terminal": False,
            "turbulence_threshold": turbulence_threshold,
            "turbulence": 0,
            "reward": 0,
            "trades": 0,
            "costs": 0,
        }

        # data
        self.df = df
        self.current_df = self.df.loc[self.env_info["day"], :]

        self.memory_buffer = {
            "total_asset_value_memory": [env_config.INITIAL_ACCOUNT_BALANCE],
            "rewards_memory": [],
        }

        # continous action_space with np.float32 from STRONG_SELL[-1], DO_NOTHING[0], STRONG_BUY[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(data_config.STOCK_DIM, ), dtype=np.float32)

        # TODO: can shares owned be fractional?
        self.observation_space = spaces.Dict({
            "account_balance":
            spaces.Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
            "adjusted_close_price":
            spaces.Box(low=0, high=np.inf, shape=(data_config.STOCK_DIM, ), dtype=np.float32),
            "shares_owned_fractional":
            spaces.Box(low=0, high=np.inf, shape=(data_config.STOCK_DIM, ), dtype=np.float32),
            # MACD: one of the most commonly used momentum indicator that identifies moving averages
            "moving_average_convergence_divergence":
            spaces.Box(low=0, high=np.inf, shape=(data_config.STOCK_DIM, ), dtype=np.float32),
            # RSI: quantifies the extent of recent price changes. If price moves around the support line, it indicates the stock is oversold, and we can perform the buy action (and vice-versa)
            "relative_strength_index ":
            spaces.Box(low=0, high=np.inf, shape=(data_config.STOCK_DIM, ), dtype=np.float32),
            # CCI: compares current price to average price over a time window to indicate a buying or selling action
            "commodity_channel_index":
            spaces.Box(low=0, high=np.inf, shape=(data_config.STOCK_DIM, ), dtype=np.float32),
            # ADX: identifies trend strength by quantifying the amount of price movement
            "average_directional_index":
            spaces.Box(low=0, high=np.inf, shape=(data_config.STOCK_DIM, ), dtype=np.float32)
        })

        self.observation = {
            "account_balance": [env_config.INITIAL_ACCOUNT_BALANCE],
            "adjusted_close_price": self.current_df.adjcp.values.tolist(),
            "shares_owned_fractional": [0] * data_config.STOCK_DIM,
            "moving_average_convergence_divergence": self.current_df.macd.values.tolist(),
            "relative_strength_index ": self.current_df.rsi.values.tolist(),
            "commodity_channel_index": self.current_df.cci.values.tolist(),
            "average_directional_index": self.current_df.adx.values.tolist()
        }

        self.seed()

    def render(self, mode='human', close=False):
        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        if self.env_info["split"] == "test" and self.env_info["training_iteration"] != 0:
            # restore previous observation in test env to simulate trading
            self.observation = self.env_info["previous_observation"]
            total_asset_value = self._get_total_assets()
        else:
            # init default state
            self.observation = {
                "account_balance": [env_config.INITIAL_ACCOUNT_BALANCE],
                "adjusted_close_price": self.current_df.adjcp.values.tolist(),
                "shares_owned_fractional": [0] * data_config.STOCK_DIM,
                "moving_average_convergence_divergence": self.current_df.macd.values.tolist(),
                "relative_strength_index ": self.current_df.rsi.values.tolist(),
                "commodity_channel_index": self.current_df.cci.values.tolist(),
                "average_directional_index": self.current_df.adx.values.tolist()
            }
            total_asset_value = env_config.INITIAL_ACCOUNT_BALANCE

        self.env_info["day"] = 0
        self.env_info["terminal"] = False
        self.env_info["turbulence"] = 0
        self.env_info["trades"] = 0
        self.env_info["costs"] = 0

        self.current_df = self.df.loc[self.env_info["day"], :]

        self.memory_buffer = {
            "total_asset_value_memory": [total_asset_value],
            "rewards_memory": [],
        }

        # pprint.pprint(self.observation, width=160, compact=True)
        return self.observation

    def step(self, actions):

        # check if we're at the max number of days
        self.env_info["terminal"] = self.env_info["day"] >= self.env_info["max_days"]

        if self.env_info["terminal"]:
            info = self._save_terminal_state()
            return self.observation, self.env_info["reward"], self.env_info["terminal"], {}

        # calculate total assets at timestep t
        total_assets_t0 = self._get_total_assets()

        # scale actions by the maximum amount of shares for each buying action
        actions = actions * env_config.HMAX_NORMALIZE
        # sell all stocks if we exceed the turbulence threshold
        if self.env_info["turbulence"] >= self.env_info["turbulence_threshold"]:
            actions = np.array([-env_config.HMAX_NORMALIZE] * data_config.STOCK_DIM)

        # iterate over actions of STRONG_SELL[-1], DO_NOTHING[0], STRONG_BUY[1]
        for idx, action in enumerate(actions):
            if action > 0:
                self._buy_stock(idx, action)
            elif action < 0:
                self._sell_stock(idx, action)

        # increase count for day
        self.env_info["day"] += 1
        self.current_df = self.df.loc[self.env_info["day"], :]
        self.env_info["turbulence"] = self.current_df['turbulence'].values[0]

        # update state based on new date
        self.observation["adjusted_close_price"] = self.current_df.adjcp.values.tolist()
        self.observation["moving_average_convergence_divergence"] = self.current_df.macd.values.tolist()
        self.observation["relative_strength_index"] = self.current_df.rsi.values.tolist()
        self.observation["commodity_channel_index"] = self.current_df.cci.values.tolist()
        self.observation["average_directional_index"] = self.current_df.adx.values.tolist()

        # calculate total assets at timestep t
        total_assets_t1 = self._get_total_assets()
        self.memory_buffer["total_asset_value_memory"].append(total_assets_t1)

        # calculate reward
        self.env_info["reward"] = total_assets_t1 - total_assets_t0
        self.memory_buffer["rewards_memory"].append(self.env_info["reward"])
        self.env_info["reward"] = self.env_info["reward"] * env_config.REWARD_SCALING

        return self.observation, self.env_info["reward"], self.env_info["terminal"], {}

    def _get_total_assets(self) -> float:
        account_balance = self.observation["account_balance"]
        adjusted_close_price = self.observation["adjusted_close_price"]
        shares_owned_fractional = self.observation["shares_owned_fractional"]
        total_assets = np.array(account_balance) + np.sum(np.array(adjusted_close_price) * np.array(shares_owned_fractional))
        return float(total_assets)

    def _buy_stock(self, index, action):
        # get values from observation
        account_balance = self.observation["account_balance"]
        adjusted_close_price = self.observation["adjusted_close_price"]
        shares_owned_fractional = self.observation["shares_owned_fractional"]

        # perform buy action based on min number of shares to purchase
        available_amount = account_balance[0] // adjusted_close_price[index]
        shares_to_buy = min(action, available_amount)

        # update balance
        account_balance -= adjusted_close_price[index] * shares_to_buy * (1 + env_config.TRANSACTION_FEE_PERCENT)
        shares_owned_fractional[index] += shares_to_buy
        self.env_info["costs"] += adjusted_close_price[index] * shares_to_buy * env_config.TRANSACTION_FEE_PERCENT
        self.env_info["trades"] += 1

        # update observation
        self.observation["account_balance"] = account_balance
        self.observation["shares_owned_fractional"] = shares_owned_fractional

    def _sell_stock(self, index, action):
        # get values from observation
        account_balance = self.observation["account_balance"]
        adjusted_close_price = self.observation["adjusted_close_price"]
        shares_owned_fractional = self.observation["shares_owned_fractional"]

        # perform buy action based a min
        shares_to_sell = min(abs(action), shares_owned_fractional[index])

        # we must own at least greater than 0 shares to sell
        if shares_owned_fractional[index] > 0:
            account_balance += adjusted_close_price[index] * shares_to_sell * (1 - env_config.TRANSACTION_FEE_PERCENT)
            shares_owned_fractional[index] -= shares_to_sell
            self.env_info["costs"] += adjusted_close_price[index] * shares_to_sell * env_config.TRANSACTION_FEE_PERCENT
            self.env_info["trades"] += 1

        # update observation
        self.observation["account_balance"] = account_balance
        self.observation["shares_owned_fractional"] = shares_owned_fractional

    def _get_sharpe_ratio(self):
        df_total_value = pd.DataFrame(self.memory_buffer["total_asset_value_memory"])
        df_total_value.columns = ['account_value']
        df_total_value['daily_return'] = df_total_value.pct_change(1)
        if self.env_info["split"] == "train":
            sharpe_ratio = np.sqrt(252) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
        elif self.env_info["split"] == "val" or self.env_info["split"] == "test":
            sharpe_ratio = np.sqrt(4) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
        return sharpe_ratio

    def _save_terminal_state(self, _DELAY=10):

        # init filenames
        total_asset_value_memory_plot_filename = os.path.join(paths_config.results_figs_dir,
                                                              f'total_asset_value_{self.env_info["split"]}_{self.env_info["training_iteration"]}.png')
        total_asset_value_memory_csv_filename = os.path.join(paths_config.results_csv_dir,
                                                             f'total_asset_value_{self.env_info["split"]}_{self.env_info["training_iteration"]}.csv')
        account_rewards_csv_filename = os.path.join(paths_config.results_csv_dir, f'rewards_{self.env_info["split"]}_{self.env_info["training_iteration"]}.csv')

        # get params for logging
        account_balance = int(self.observation["account_balance"][0])
        total_assets = int(self._get_total_assets())
        sharpe_ratio = self._get_sharpe_ratio()

        # save total asset value as csv
        df_total_value = pd.DataFrame(self.memory_buffer["total_asset_value_memory"])
        df_total_value.to_csv(f'{total_asset_value_memory_csv_filename}')

        # save rewards as csv
        df_rewards = pd.DataFrame(self.memory_buffer["rewards_memory"])
        df_rewards.to_csv(f'{account_rewards_csv_filename}')

        # get params from env info
        # pprint.pprint(self.env_info)
        reward = self.env_info["reward"]
        trades = self.env_info["trades"]
        costs = self.env_info["costs"]

        # plot total asset value
        fig = plt.figure(figsize=(10, 10))
        plt.style.use("fivethirtyeight")
        title = f'{self.env_info["split"]} env \nAccount Balance: $ {account_balance} \nTotal Asset Value: $ {total_assets} \nSharpe Ratio: {sharpe_ratio:.2f}'
        fig.suptitle(title, fontsize=14)
        plt.ylabel('Asset Memory [$]', fontsize=12)
        plt.xlabel('Window [months]', fontsize=12)
        # plot asset memory
        plt.plot(self.memory_buffer["total_asset_value_memory"])
        # save figure
        plt.savefig(f'{total_asset_value_memory_plot_filename}')

        # print outputs to user if we are not training
        if self.env_info["split"] == "val" or self.env_info["split"] == "test":
            # TODO: print self.env_info
            print(f"account_balance: ${account_balance:.2f}")
            print(f"total_assets: ${total_assets:.2f}")
            # print(f"rewards [change in total asset value]: average: ${float(df_rewards.avg()):.2f}")
            print(
                f"rewards [change in total asset value]: min: ${float(df_rewards.min()):.2f}, max: ${float(df_rewards.max()):.2f}, std: ${float(df_rewards.std()):.2f}"
            )
            # # show P&L
            # plt.draw()
            # plt.pause(_DELAY)
            # time.sleep(_DELAY)
        plt.close()
