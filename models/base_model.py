from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self, opt, num_state, num_action, gamma=0.99):
        self.opt = opt
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma  # 割引率
    
    @abstractmethod
    def train_per_one_episode(self, episode, observation, env):
        pass
    
    # Q値が最大の行動を選択
    @abstractmethod
    def get_greedy_action(self, state):
        pass
    
    # 行動を選択
    def get_action(self, state, episode):
        pass