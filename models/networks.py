import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

class PolicyNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_action)
    
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        action_prob = F.softmax(self.fc3(h), dim=-1)
        return action_prob

# shared actor & critic network
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2a = nn.Linear(hidden_size, num_action)
        self.fc2c = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h = F.elu(self.fc1(x))
        action_prob = F.softmax(self.fc2a(h), dim=-1)
        state_value = self.fc2c(h)
        return action_prob, state_value