import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10
from tqdm import tqdm
import supersuit as ss
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, action_dim, device=device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device)
        self.fc = None
        self.actor_head = nn.Linear(128, action_dim).to(device)
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
        # Add batch dimension if input is a single observation
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension: [1, 512, 512, 3]

        # Permute dimensions for Conv2D compatibility
        x = x.permute(0, 3, 1, 2).to(self.device)  # [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_value = self.critic_head(x)
        return action_probs, state_value



def calculate_returns_and_advantages(rewards, values, dones, gamma, gae_lambda):
    returns = []
    advantages = []
    gae = 0
    R = 0
    for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        R = reward + gamma * R * (1 - done)
        returns.insert(0, R)
        delta = reward + gamma * (0 if done else values[0]) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        advantages.insert(0, gae)
    return torch.tensor(returns).to(device), torch.tensor(advantages).to(device)


def ppo_update(model, optimizer, D, clip_param, value_coef, entropy_coef, minibatch_size, epochs):
    actor_losses = []
    critic_losses = []
    entropy_losses = []
    for _ in range(epochs):
        np.random.shuffle(D)
        for i in range(0, len(D), minibatch_size):
            minibatch = D[i:i + minibatch_size]
            obs, actions, old_log_probs, returns, advantages = zip(*minibatch)

            obs = torch.cat(obs).to(device)
            actions = torch.cat(actions).to(device)
            old_log_probs = torch.cat(old_log_probs).to(device)
            returns = torch.cat(returns).to(device)
            advantages = torch.cat(advantages).to(device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            action_probs, values = model(obs)
            values = values.squeeze()
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO Clipped Objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy.item())

    return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)


def plot_everything(actor_losses, critic_losses, entropy_losses, episode_rewards):
    # Plotting the results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(actor_losses, label='Actor Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Actor Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Critic Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(entropy_losses, label='Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Entropy Loss')
    plt.savefig("ppo_losses.png")

    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.savefig("ppo.png")

# Main Training Loop
stepmax = 10
batch_size = 16 #64
gamma = 0.99
gae_lambda = 0.95
minibatch_size = 4 #16
ppo_epochs = 4
clip_param = 0.2
lr = 3e-4
value_coef = 0.5
entropy_coef = 0.01

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=15,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
)

# Apply SuperSuit wrappers to make observations smaller and greyscale
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.color_reduction_v0(env, mode='full')

observations, infos = env.reset()
model = ActorCritic(env.action_space('archer_0').n).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

actor_losses, critic_losses, entropy_losses = [], [], []
episode_rewards = []

trajectory_size = 100  # Desired size of each trajectory
step = 0
with tqdm(total=stepmax) as pbar:
    while step < stepmax:
        D = []
        batch_rewards = []
        for _ in range(batch_size):
            trajectory = []
            observations, infos = env.reset()
            steps_in_trajectory = 0
            done = False

            while steps_in_trajectory < trajectory_size:
                actions = {}
                for agent in env.agents:
                    obs = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0).to(device)
                    action_probs, value = model(obs)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    actions[agent] = action.item()

                    trajectory.append({
                        "obs": obs,
                        "action": action,
                        "value": value.item(),
                        "log_prob": dist.log_prob(action),
                    })
                    steps_in_trajectory += 1

                    # Stop early if we've reached the desired trajectory size
                    if steps_in_trajectory >= trajectory_size:
                        break

                observations, rewards, terminations, truncations, infos = env.step(actions)
                batch_rewards.append(sum(rewards.values()))

                # If all agents are done, reset the environment
                if all(terminations.values()) or all(truncations.values()):
                    observations, infos = env.reset()

            # Assign rewards to the trajectory
            for agent, traj_entry in zip(env.agents, trajectory):
                traj_entry["reward"] = rewards.get(agent, 0)
                
            D.extend(trajectory)


        episode_rewards.append(np.mean(batch_rewards))
        # Process data for PPO update
        ppo_loss = ppo_update(model, optimizer, D, clip_param, value_coef, entropy_coef, minibatch_size, ppo_epochs)
        pbar.update(1)

        if step % 2 == 0:
            plot_everything(actor_losses, critic_losses, entropy_losses, episode_rewards)
