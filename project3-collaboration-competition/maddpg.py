from ddpg import Agent
import torch
import random
from collections import namedtuple, deque
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG(object):
    def __init__(self, num_agents, state_size, action_size, seed=None, buffer_size=int(1e5), batch_size=128,
                 gamma=0.99, minsamples_before_train=5000):
        """Maintain multiple agent using the MADDPG algorithm.

        Parameters
        ----------
        num_agents: int
            Number of agents
        state_size: int
            Size of the state space
        action_size: int
            Size of the action space
        seed: int
            Random seed for the weights of each agent and action noise
        buffer_size: int
            Size of the replay buffer
        batch_size: int
            Batch size for the optimizer
        gamma: float
            Discount factor
        minsamples_before_train: int
            The minimum number of samples in the buffer before starting the training
        """
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.minsamples_before_train = minsamples_before_train

        # Multiple agents are initialized
        self.agents = [Agent(state_size, action_size, seed) for _ in range(self.num_agents)]

        # Shared replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
    
    def act(self, states, add_noise=True):
        """Ask each agent to perform an action"""
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions
    
    def reset(self):
        """Reset the noise for each agent"""
        for agent in self.agents:
            agent.reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        """Take a step in time for each agent by saving their experiences to the buffer and learning"""

        # Save trajectories to Replay buffer
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
        if len(self.memory) > self.minsamples_before_train:
            for _ in range(self.num_agents):
                experience = self.memory.sample()
                self.learn(experience, gamma=self.gamma)
    
    def learn(self, experiences, gamma=0.99):
        """Learn for each agent"""
        for agent in self.agents:
            agent.learn(experiences, gamma)
            
    def saveModelParams(self):
        """Saves the parameters of the actor and critic for each agent"""
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"critic_agent_{i}.pth")


class ReplayBuffer(object):
    """Experience replay buffer"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initializes the replay buffer

        Parameters
        ----------
        action_size: int
            Size of the action space
        buffer_size: int
            Size of the replay buffer
        batch_size: int
            Batch size
        seed: int
            Seed for the random sampling from the buffer
        """
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.replay_memory = deque(maxlen=buffer_size)  # Experience replay memory object
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state",
                                                                "done"])  # standard S,A,R,S',done

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience_tuple = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(experience_tuple)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.replay_memory)
