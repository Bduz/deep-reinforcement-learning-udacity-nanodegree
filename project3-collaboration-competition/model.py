import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1=128, fc2=64):
        """Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)

        self.fc2 = nn.Linear(fc1, fc2) 
        self.bn2 = nn.BatchNorm1d(fc2)

        self.fc3 = nn.Linear(fc2, action_size)  # µ(s|θ) {Deterministic policy}

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = state

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        mu = torch.tanh(x)
        return mu


class Critic(nn.Module):
    """Critic (Value) Model, V(pi) as Q(s, a|θ)"""
    def __init__(self, state_size, action_size, seed, fc1=128, fc2=64):
        """Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fcs1_units (int): Number of nodes in the first hidden layer
        fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)

        self.fc2 = nn.Linear(fc1 + action_size, fc2)

        self.fc3 = nn.Linear(fc2, 1)  # Q-value

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)

        x = self.fc1(state)
        x = F.relu(x)
        x = self.bn1(x)

        x = torch.cat((x, action), dim=1)
        x = self.fc2(x) 
        x = F.relu(x)

        value = self.fc3(x)
        return value
