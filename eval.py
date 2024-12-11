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
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, stride=2)  # Output: 41x41x16
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)  # Output: 20x20x32
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)  # Output: 9x9x32

        # Flattened size: 32 * 9 * 9 = 2592
        self.fc = nn.Linear(2592, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

        self.to(device)

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        # Add to ActorCritic __init__ after self.to(device):
        self.apply(init_weights)

    def forward(self, x):
        # Convert to tensor if not already one
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Move to device if needed
        if x.device != device:
            x = x.to(device)

        # Handle observation shape normalization
        if len(x.shape) == 3:  # Single observation (height, width, channels)
            x = x.unsqueeze(0)  # Add batch dimension -> (1, height, width, channels)

        # Ensure channel-first format (batch_size, channels, height, width)
        if x.shape[-1] == 4:  # If channels are last (e.g., NHWC format)
            x = x.permute(
                0, 3, 1, 2
            )  # Reorder to (batch_size, channels, height, width)

        # Pass through the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))

        # Compute outputs
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


class ActorCriticOld(nn.Module):
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
                nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain("relu"))
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
            x = x.unsqueeze(0).unsqueeze(
                0
            )  # Add batch and channel dims -> (1, 1, 84, 84)
        elif len(x.shape) == 3:  # If input is (1, 84, 84) or (3, 84, 84)
            if x.shape[0] == 3:  # If RGB
                x = x.mean(dim=0, keepdim=True).unsqueeze(
                    0
                )  # Convert to grayscale and add batch
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
    render_mode=None,
    spawn_rate=15,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=30,
)

env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize to 84x84
env = ss.color_reduction_v0(env, mode="full")  # Grayscale
env = ss.frame_stack_v2(env, stack_size=4)  # Stack 4 frames

# Load the trained model weights
# model_path = 'models/run6/mappo_model_episode_8000_1733282084.905826.pth'
model_path = "models/run12/mappo_model_episode_8000_1733609120.1231153.pth"
model = ActorCritic(action_dim=env.action_space("archer_0").n)
model.load_state_dict(torch.load(model_path, map_location=device))


model.eval()
all_rewards = []
with torch.no_grad():
    for game in tqdm(range(20)):
        observations, infos = env.reset()
        total_reward = 0
        for step in tqdm(range(1000), leave=False):
            if not env.agents:
                break
            actions = {}
            for agent in env.agents:
                eval_obs = torch.FloatTensor(observations[agent]).to(device)
                eval_action_probs, _ = model(eval_obs)
                eval_dist = Categorical(eval_action_probs)
                eval_action = eval_dist.sample()
                actions[agent] = eval_action.item()
            observations, rewards, terminations, truncations, infos = env.step(actions)
            env.render()
            total_reward += sum(rewards.values())
        all_rewards.append(total_reward)

# Calculate statistics
mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)

# Plotting the rewards
plt.figure(figsize=(6, 6))
plt.plot(range(1, 21), all_rewards, marker="o", label="Total Reward per Game")
plt.axhline(
    mean_reward, color="r", linestyle="--", label=f"Mean Reward: {mean_reward:.2f}"
)
plt.fill_between(
    range(1, 21),
    mean_reward - std_reward,
    mean_reward + std_reward,
    color="r",
    alpha=0.2,
    label="±1 Std Dev",
)
plt.xlabel("Game")
plt.ylabel("Total Rewards")
plt.title("Total Rewards per Game")
plt.legend()
plt.grid(True)
plt.savefig("rewards_run12_a.png")
