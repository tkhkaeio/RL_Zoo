from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self, opt, num_state, num_action, gamma=0.99):
        self.opt = opt
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma  # discount rate
    
    @abstractmethod
    def train_per_one_episode(self, episode, observation, env):
        pass
    
    # choose the action with the highest Q-value
    @abstractmethod
    def get_greedy_action(self, state):
        pass
    
    # choose an action
    def get_action(self, state, episode):
        pass