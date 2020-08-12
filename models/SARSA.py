import sys
import numpy as np
from .base_model import BaseModel
sys.path.append("../utils ")
from utils.helpers import bins, discretize_state

class SARSAAgent(BaseModel):
    def __init__(self, opt, num_state, num_action, gamma=0.99, alpha=0.5, max_initial_q=0.1):
        BaseModel.__init__(self, opt, num_state, num_action, gamma)
        self.num_discretize = opt.num_discretize
        self.alpha = alpha  # 学習率
        self.qtable = np.random.uniform(low=-max_initial_q, high=max_initial_q, size=(self.num_discretize ** self.num_state, self.num_action))
        
    def train_per_one_episode(self, episode, observation, env):
        state = discretize_state(observation, self.num_discretize)  # 観測の離散化（状態のインデックスを取得）
        action = self.get_action(state, episode)  #  行動を選択
        episode_reward = 0
        for t in range(self.opt.max_steps):
            observation, reward, done, _ = env.step(action)
            # もしエピソードの途中で終了してしまったらペナルティを加える
            if done and t < self.opt.max_steps - 1:
                reward = - self.opt.penalty
            episode_reward += reward
            next_state = discretize_state(observation, self.num_discretize)
            next_action = self.get_action(next_state, episode)
            self.update_qtable(state, action, reward, next_state, next_action)  # Q値の表を更新
            state, action = next_state, next_action
            if done:
                break
        return episode_reward
    
    # Qテーブルを更新
    def update_qtable(self, state, action, reward, next_state, next_action):
        self.qtable[state, action] += self.alpha * (reward + self.gamma * self.qtable[next_state, next_action] - self.qtable[state, action])
    
    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        action = np.argmax(self.qtable[state])
        return action
    
    # ε-greedyに行動を選択 (default)
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定
        if epsilon <= np.random.uniform(0,1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action