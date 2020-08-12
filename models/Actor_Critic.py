import numpy as np
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from .base_model import BaseModel
from .networks import ActorCriticNetwork


class ActorCriticAgent(BaseModel):
    def __init__(self, opt, num_state, num_action, gamma=0.99, lr=0.001, beta1=0.9, beta2=0.999):
        BaseModel.__init__(self, opt, num_state, num_action, gamma)
        self.ac_net = ActorCriticNetwork(num_state, num_action)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr, betas=(beta1, beta2))
        self.memory = []  # （報酬，行動選択確率，状態価値）のtupleをlistで保存
    
    def train_per_one_episode(self, episode, observation, env):
        state = observation # use continuous state
        episode_reward = 0
        for t in range(self.opt.max_steps):
            action, prob, state_value = self.get_action(state)  #  行動を選択
            next_state, reward, done, _ = env.step(action)
            # # もしエピソードの途中で終了してしまったらペナルティを加える
            # if done and t < self.opt.max_steps - 1:
            #     reward = - penalty
            episode_reward += reward
            self.add_memory(reward, prob, state_value)
            state = next_state
            if done:
                self.update_policy()
                self.reset_memory()
                break
        return episode_reward
        
    # 方策を更新
    def update_policy(self):
        R = 0
        actor_loss = 0
        critic_loss = 0
        for r, prob, v in self.memory[::-1]:
            R = r + self.gamma * R
            advantage = R - v
            actor_loss -= torch.log(prob) * advantage
            critic_loss += F.smooth_l1_loss(v, torch.tensor(R))
        actor_loss = actor_loss/len(self.memory)
        critic_loss = critic_loss/len(self.memory)
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
    
    # softmaxの出力が最も大きい行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob, _ = self.ac_net(state_tensor.data)
        action = torch.argmax(action_prob.squeeze().data).item()
        return action
    
    # カテゴリカル分布からサンプリングして行動を選択
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob, state_value = self.ac_net(state_tensor.data)
        action_prob, state_value = action_prob.squeeze(), state_value.squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action], state_value
    
    def add_memory(self, r, prob, v):
        self.memory.append((r, prob, v))
    
    def reset_memory(self):
        self.memory = []