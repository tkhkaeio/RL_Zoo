import numpy as np
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_model import BaseModel
from .networks import QNetwork
from .modules import ReplayBuffer



class DQNAgent(BaseModel):
    def __init__(self, opt, num_state, num_action, gamma=0.99, lr=0.001, beta1=0.9, beta2=0.999, batch_size=32, memory_size=50000):
        BaseModel.__init__(self, opt, num_state, num_action, gamma)
        self.batch_size = batch_size  # Q関数の更新に用いる遷移の数
        self.q_net = QNetwork(num_state, num_action)
        self.target_qnet = copy.deepcopy(self.q_net)  # ターゲットネットワーク
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, betas=(beta1, beta2))
        self.replay_buffer = ReplayBuffer(memory_size)

    # 最初にreplay bufferにランダムな行動をしたときのデータを入れる
    def init_replay_buffer(self, env):
        state = env.reset()
        for step in range(self.opt.initial_memory_size):
            action = env.action_space.sample() # ランダムに行動を選択        
            next_state, reward, done, _ = env.step(action)
            transition = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'done': int(done)
            }
            self.replay_buffer.append(transition)
            state = env.reset() if done else next_state

    def train_per_one_episode(self, episode, observation, env):
        state = observation # use continuous state
        episode_reward = 0
        for t in range(opt.max_steps):
            action = agent.get_action(state).data.numpy()  #  行動を選択
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            transition = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'done': int(done)
            }
            agent.replay_buffer.append(transition)
            agent.update()  # actorとcriticを更新
            state = next_state
            if done:
                break
        return episode_reward
            
        
    # Q関数を更新
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.q_net(torch.tensor(batch["states"], dtype=torch.float))
        targetq = copy.deepcopy(q.data.numpy())
        # maxQの計算
        maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"],dtype=torch.float)), dim=1).values
        # Q値が最大の行動だけQ値を更新（最大ではない行動のQ値はqとの2乗誤差が0になる）
        for i in range(self.batch_size):
            # 終端状態の場合はmaxQを0にしておくと学習が安定します（ヒント：maxq[i] * (not batch["dones"][i])）
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma * maxq[i] * (not batch["dones"][i])
            
        self.optimizer.zero_grad()
        # lossとしてMSEを利用
        loss = self.criterion(q, torch.tensor(targetq))
        loss.backward()
        self.optimizer.step()
        # ターゲットネットワークのパラメータを更新
        self.target_qnet = copy.deepcopy(self.q_net)
    
    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action = torch.argmax(self.q_net(state_tensor).data).item()
        return action
    
    # ε-greedyに行動を選択 (default)
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定
        if epsilon <= np.random.uniform(0,1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action