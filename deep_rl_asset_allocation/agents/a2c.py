import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

LEARNING_RATE = 3e-2
SEED = 543
GAMMA = 0.999
LOG_INTERVAL = 10
CLIP_GRADIENT = 0.15

SavedAction = namedtuple('SavedAction', ['log_prob', 'state_value'])


class ContinousPolicy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, n_obs, n_actions, hidden_layers=128):
        super(ContinousPolicy, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(n_obs, hidden_layers),
            nn.ReLU(),
        )
        self.action_head_mu = nn.Sequential(
            nn.Linear(hidden_layers, n_actions),
            nn.Tanh(),
        )
        self.action_head_var = nn.Sequential(
            nn.Linear(hidden_layers, n_actions),
            nn.Softplus(),
        )
        self.state_value_head = nn.Linear(hidden_layers, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        base_out = self.base(x)

        # actor (i.e. the policy): choses action to take
        # from state s_t by returning probability of each action
        mu = self.action_head_mu(base_out)
        var = self.action_head_var(base_out)

        # critic: evaluates an estimate of how many rewards
        # they expect to get from that point onwards (i.e. the state value)
        state_value = self.state_value_head(base_out)

        return mu, var, state_value


class A2C:

    def __init__(self, env, policy):

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        # seed for more deterministic training results
        self.env.seed(SEED)
        torch.manual_seed(SEED)

        # init policy and optimizer for training
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.eps = np.finfo(np.float32).eps.item()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def _calc_logprob(self, mu_v, var_v, actions_v):
        # ensure lower bound is not zero
        var_eps = var_v.clamp(min=1e-3)
        p1 = -((mu_v - actions_v)**2) / (2 * var_eps)
        p2 = -torch.log(torch.sqrt(2 * math.pi * var_eps))
        log_prob = p1 + p2
        return log_prob

    def _select_action(self, state):
        state = torch.from_numpy(state).float()
        mu_v, var_v, state_value = self.policy(state)

        # create a guassian distribution over the list of probabilities of actions
        actions = torch.normal(mu_v, var_v)

        # get log probability
        log_prob = self._calc_logprob(mu_v, var_v, actions)
        # save to action buffer
        self.saved_actions.append(SavedAction(log_prob, state_value))

        # the action to take (left or right)
        actions = actions.detach().cpu().numpy()
        actions = np.clip(actions, -1, 1)
        return actions

    def _update_policy(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        saved_actions = self.saved_actions
        actor_loss = []  # list to save actor (policy) loss
        critic_loss = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        R = 0
        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, state_value), R in zip(saved_actions, returns):
            # check for gradients
            # print(f"State Value Gradient: {state_value.grad}")
            # print(f"Log Probability Gradient: {log_prob.grad}")

            advantage = R - state_value.item()

            # calculate actor (policy) loss
            actor_loss.append(-log_prob * advantage)

            # calculate critic (state value) loss using L1 smooth loss
            critic_loss.append(F.smooth_l1_loss(state_value, torch.tensor([R])))

        # sum up all the values of policy_losses and value_losses
        # TODO: include entropy in the loss
        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        # print(f"Loss Data: {loss.data}")
        # print(f"Loss Gradient: {loss.grad}")

        # reset gradients
        self.optimizer.zero_grad()
        # perform backprop
        loss.backward()
        # print(f"Loss Gradient: {loss.grad}")
        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), CLIP_GRADIENT)
        # update weights
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def learn(self, num_steps=1000, total_timesteps=25000):
        running_reward = 10

        idx_episode = 0
        idx_timestep = 0
        while idx_timestep < total_timesteps:
            idx_episode += 1

            # reset environment and episode reward
            state = self.env.reset()
            episode_reward = 0

            # TODO: for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            # for t in range(1, 10000):
            for t in range(1, num_steps):
                idx_timestep += 1

                # select action from policy
                action = self._select_action(state)

                # take the action
                state, reward, done, info = self.env.step(action)

                self.rewards.append(reward)
                episode_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # perform backprop
            self._update_policy()

            # log results
            # if i_episode % LOG_INTERVAL == 0:
            print(f'Timestep: {idx_timestep},\t Episode: {idx_episode},\t Episode reward: {episode_reward:.2f},\t Running reward: {running_reward:.2f}')
            # print(f'Episode: {i_episode + 1},\t Last reward: {ep_reward:.2f},\t Average reward: {running_reward:.2f}')

        return self.policy

    def predict(self, observation):
        # select action from policy
        action = self._select_action(observation)
        return action, observation

    def save(self, filename="actor_critic.pth"):
        checkpoint = {}
        checkpoint["policy"] = self.policy.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        # checkpoint["epochs"] = epochs

        torch.save(checkpoint, filename)
        print(f'saved model to {filename} ..\n')
