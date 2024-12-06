import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10
from tqdm import tqdm
import supersuit as ss
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    
    def __init__(self, action_dim):
        super().__init__()
        # Input: 84x84 grayscale
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)  # Output: 20x20x32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # Output: 9x9x64
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # Output: 7x7x64
        
        # Calculate flattened size: 7 * 7 * 64 = 3136
        self.fc = nn.Linear(3136, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        
        self.to(device)
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        self.apply(init_weights)

    def forward(self, x):
        # Handle observation preprocessing
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Move to device if not already there
        if x.device != device:
            x = x.to(device)
        
        # Ensure proper dimensions
        if len(x.shape) == 2:  # If input is (84, 84)
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims -> (1, 1, 84, 84)
        elif len(x.shape) == 3:  # If input is (1, 84, 84) or (3, 84, 84)
            if x.shape[0] == 3:  # If RGB
                x = x.mean(dim=0, keepdim=True).unsqueeze(0)  # Convert to grayscale and add batch
            else:
                x = x.unsqueeze(0)  # Just add batch dimension
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, 3136)
        x = F.relu(self.fc(x))
        
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value
    
    
env = knights_archers_zombies_v10.parallel_env(
        render_mode="human",
        spawn_rate=15,
        vector_state=False,
        num_archers=2,
        num_knights=2,
        max_zombies=30,
    )
    
env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize to 84x84
env = ss.color_reduction_v0(env, mode='full')  # Grayscale

# Load the trained model weights
model_path = '/home/anish/Code/MAPPO-KAZ/mappo_model_episode_9999_1733306990.2326434.pth'
model = ActorCritic(action_dim=env.action_space("archer_0").n)
model.load_state_dict(torch.load(model_path, map_location=device))

# Run a game with the loaded policy
observations, infos = env.reset()
while env.agents:
    actions = {}
    for agent in env.agents:
        obs = observations[agent]
        action_probs, _ = model(obs)
        action = torch.argmax(action_probs).item()
        actions[agent] = action
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # Optionally, render the environment
    # env.render()

env.close()