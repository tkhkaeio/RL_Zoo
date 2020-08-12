import numpy as np
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from .base_model import BaseModel
from .networks import PolicyNetwork

class REINFORCEAgent(BaseModel):
    def __init__(self, opt, num_state, num_action, gamma=0.99, lr=0.001, beta1=0.9, beta2=0.999, use_baseline=False):
        BaseModel.__init__(self, opt, num_state, num_action, gamma)
        self.pi_net = PolicyNetwork(num_state, num_action)
        self.optimizer = optim.Adam(self.pi_net.parameters(), lr=lr, betas=(beta1, beta2))
        self.memory = []  # 報酬とそのときの行動選択確率のtupleをlistで保存
        self.sum_reward = 0
        self.use_baseline = use_baseline
        if self.use_baseline:
            opt.name += "_with_baseline"
    
    def train_per_one_episode(self, episode, observation, env):
        state = observation # use continuous state
        episode_reward = 0
        for t in range(self.opt.max_steps):
            action, prob = self.get_action(state)  #  行動を選択
            next_state, reward, done, _ = env.step(action)
            # # もしエピソードの途中で終了してしまったらペナルティを加える
            # if done and t < self.opt.max_steps - 1:
            #     reward = - penalty
            episode_reward += reward
            self.add_memory(reward, prob)
            state = next_state
            if done:
                if self.use_baseline:
                    self.update_policy_with_baseline()
                else:
                    self.update_policy()
                self.reset_memory()
                break
        return episode_reward

    # 方策を更新
    def update_policy(self):
        R = 0
        loss = 0
        # エピソード内の各ステップの収益を後ろから計算
        for r, prob in self.memory[::-1]:
            R = r + self.gamma * R
            loss -= torch.log(prob) * R
        loss = loss/len(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    # 方策を更新
    def update_policy_with_baseline(self):
        R = 0
        loss = 0
        b  = self.sum_reward / len(self.memory) # average reward
        # エピソード内の各ステップの収益を後ろから計算
        for r, prob in self.memory[::-1]:
            R = r + self.gamma * R
            f_pi  = R - b
            loss -= torch.log(prob) * f_pi
        loss = loss/len(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    # softmaxの出力が最も大きい行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob = self.pi_net(state_tensor.data).squeeze()
        action = torch.argmax(action_prob.data).item()
        return action

    # カテゴリカル分布からサンプリングして行動を選択
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob = self.pi_net(state_tensor.data).squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action]
    
    def add_memory(self, r, prob):
        self.memory.append((r, prob))
        self.sum_reward += r
    
    def reset_memory(self):
        self.memory = []