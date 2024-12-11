import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from pettingzoo.butterfly import knights_archers_zombies_v10
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import prep.PopArt as PopArt
# Append step data to trajectories

# todo: compute advantage estimate using gae on trajectories and popart
# todo: compute reward-to-go on trajectories and normalize
# todo: split trajectory into chunks of length chunk_size
# todo: update D with chunks
# todo: update actor and critic using D
# adam optimizer update


class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layer
        self.fc = nn.Linear(64 * 61 * 61, 128)
        # Output layer
        self.actor_head = nn.Linear(128, action_dim)

    def forward(self, x, hidden_state=None):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        # Output layer
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        return action_probs, hidden_state


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layer
        self.fc = nn.Linear(64 * 61 * 61, 128)
        # Output layer
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x, hidden_state=None):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        # Output layer
        state_value = self.critic_head(x)
        return state_value, hidden_state


env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=20,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=30,
)
env.reset()


alpha = 0.1
lr = 0.0001

actor = Actor(env.action_spaces["archer_0"].n)
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

lr = 0.0001
stepmax = 100
batch_size = 2
lstm_hidden_state_size = 128
trajectory_size = 10
chunk_size = 2

agents = env.agents
observation_space = env.observation_space(agents[0])
action_space = env.action_space(agents[0])
input_channels = 3
action_dim = action_space.n

step = 0
pbar = tqdm(total=stepmax, desc="Timesteps")
while step <= stepmax:
    D = []  # replay buffer
    with torch.no_grad():
        # calculate trajectory for each batch
        for _ in range(batch_size):
            trajectory = []
            for _ in range(trajectory_size):
                step_data = {}
                actions = {}
                for i, agent in enumerate(agents):
                    obs = env.observe(agent)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_dist = Categorical(actor(obs))
                    action = action_dist.sample()
                    actions[agent] = action.item()

                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
