import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode="human",
    spawn_rate=20,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=30,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, action_dim, device=device):
        super(Actor, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device)
        self.fc = None
        self.actor_head = nn.Linear(128, action_dim).to(device)
        self._initialize_fc()

    def _initialize_fc(self):
        dummy_input = torch.zeros(1, 3, 512, 512).to(self.device)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        input_size = x.size(1)
        self.fc = nn.Linear(input_size, 128).to(self.device)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, device=device):
        super(Critic, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device)
        self.fc = None
        self.critic_head = nn.Linear(128, 1).to(device)
        self._initialize_fc()

    def _initialize_fc(self):
        dummy_input = torch.zeros(1, 3, 512, 512).to(self.device)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        input_size = x.size(1)
        self.fc = nn.Linear(input_size, 128).to(self.device)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        state_value = self.critic_head(x)
        return state_value

actor = Actor(env.action_spaces["archer_0"].n).to(device)
critic = Critic().to(device)

# Load the trained weights
actor.load_state_dict(torch.load("out/actor_weights.pth"))
critic.load_state_dict(torch.load("out/critic_weights.pth"))


num_games = 50
rewards_per_game = []

from tqdm import tqdm

for game in tqdm(range(num_games)):
    observations, infos = env.reset()
    total_reward = 0
    while env.agents:
        actions = {}
        for agent in env.agents:
            obs = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = actor(obs)
            dist = Categorical(action_probs)
            action = dist.sample().item()
            actions[agent] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        for reward in rewards.values():
            if reward == 1:
                print(reward)
            total_reward += reward
        # total_reward += sum(rewards.values())

    rewards_per_game.append(total_reward)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(range(num_games), rewards_per_game, marker='o')
plt.xlabel('Game')
plt.ylabel('Total Reward')
plt.title('Total Reward per Game')

plt.tight_layout()
plt.savefig('out/eval_results.png')