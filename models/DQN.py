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
        self.batch_size = batch_size  # the number of transitions to use for updating the Q function
        self.q_net = QNetwork(num_state, num_action)
        self.target_q_net = copy.deepcopy(self.q_net)  # target network
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, betas=(beta1, beta2))
        self.replay_buffer = ReplayBuffer(memory_size)

    # put the data from the first random action in the replay buffer
    def init_replay_buffer(self, env):
        state = env.reset()
        for step in range(self.opt.initial_memory_size):
            action = env.action_space.sample() # choose an action randomly  
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
        for t in range(self.opt.max_steps):
            action = self.get_action(state, episode)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            transition = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'done': int(done)
            }
            self.replay_buffer.append(transition)
            self.update_q()  # update Q-function
            state = next_state
            if done:
                break
        return episode_reward
            
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.q_net(torch.tensor(batch["states"], dtype=torch.float))
        targetq = copy.deepcopy(q.data.numpy())
        # calculate maxQ
        maxq = torch.max(self.target_q_net(torch.tensor(batch["next_states"],dtype=torch.float)), dim=1).values
        # update the Q-value only for actions with the highest Q-value.
        for i in range(self.batch_size):
            # for the terminal state, setting maxQ to 0 will stabilize the training
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma * maxq[i] * (not batch["dones"][i])
            
        self.optimizer.zero_grad()
        loss = self.criterion(q, torch.tensor(targetq))
        loss.backward()
        self.optimizer.step()
        # update the parameters of the target network
        self.target_q_net = copy.deepcopy(self.q_net)
    
    # choose the action with the highest Q-value
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action = torch.argmax(self.q_net(state_tensor).data).item()
        return action
    
    # choose an action to ε-greedy
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))
        if epsilon <= np.random.uniform(0,1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action