import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10
from tqdm import tqdm
import matplotlib.pyplot as plt
from prep.PopArt import PopArt

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


# Plotting the losses and rewards
def plot_losses_and_rewards(
    actor_losses, critic_losses, episode_rewards, iter=None, save=True
):
    # Clip outliers for better visualization
    actor_losses = np.clip(
        actor_losses, a_min=None, a_max=np.percentile(actor_losses, 95)
    )
    critic_losses = np.clip(
        critic_losses, a_min=None, a_max=np.percentile(critic_losses, 95)
    )

    plt.figure(figsize=(12, 5))

    # Plot actor and critic losses
    plt.subplot(1, 2, 1)
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Actor and Critic Loss")
    plt.legend()

    # Plot episode rewards
    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards, label="Episode Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Episode Rewards")
    plt.legend()

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f'out/losses_rewards{"" if iter is None else iter}.png')


# Training Loop
stepmax = 500
batch_size = 16
gamma = 0.99
gae_lambda = 0.95
minibatch_count = 4
step = 0
lr = 0.00001
T = 500
k = 20
beta = 0.0003
epsilon = 1e-5

# Initialize environment and models
env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=15,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=40,
    max_cycles=500,
)
observations, infos = env.reset()

# actor = Actor(env.action_spaces['archer_0'].n).to(device)
actor = Actor(env.action_space("archer_0").n).to(device)
critic = Critic().to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

actor_losses = []
critic_losses = []

episode_rewards = []

popart_returns = PopArt(beta, epsilon)
popart_advantages = PopArt(beta, epsilon)

# Outer loop with tqdm
with tqdm(total=stepmax, desc="Training Steps") as pbar:
    while step <= stepmax:
        D = []
        # Inner loop to collect trajectories
        batch_rewards = []
        for _ in range(batch_size):
            trajectory = []
            env.reset()
            episode_reward = 0
            for t in range(T):
                actions = {}
                for agent in env.agents:
                    if agent not in observations:
                        obs = (
                            torch.zeros((512, 512, 3), dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                    else:
                        obs = (
                            torch.tensor(observations[agent], dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                    action_probs = actor(obs)
                    if torch.isnan(action_probs).any():
                        break
                    action_dist = Categorical(action_probs)
                    action = action_dist.sample()
                    actions[agent] = action.item()

                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
                rewards = {
                    agent: reward * 10 if reward > 0 else -0.1
                    for agent, reward in rewards.items()
                }
                episode_reward += sum(rewards.values())
                for agent in env.agents:
                    trajectory.append(
                        {
                            "obs": observations[agent],
                            "action": actions[agent],
                            "reward": rewards[agent],
                            "done": terminations[agent] or truncations[agent],
                        }
                    )
                if all(terminations.values()) or all(truncations.values()):
                    break
            # Changed: Append only k steps and calculate GAE here
            k = min(k, len(trajectory))  # Define k appropriately
            for step_idx in range(0, len(trajectory), k):
                chunk = trajectory[step_idx : step_idx + k]
                rewards = [step["reward"] for step in chunk]
                dones = [step["done"] for step in chunk]

                # Calculate returns
                returns = []
                R = 0
                for reward, done in zip(reversed(rewards), reversed(dones)):
                    R = reward + gamma * R * (1 - done)
                    returns.insert(0, R)

                # Calculate advantages using GAE
                advantages = []
                gae = 0
                for reward, done, value in zip(
                    reversed(rewards), reversed(dones), reversed(returns)
                ):
                    delta = reward + gamma * (0 if done else returns[0]) - value
                    gae = delta + gamma * gae_lambda * (1 - done) * gae
                    advantages.insert(0, gae)

                # PopArt normalization (implement PopArt if needed)
                returns = torch.tensor(returns).to(device)
                advantages = torch.tensor(advantages).to(device)

                D.append(
                    {
                        "obs": [step["obs"] for step in chunk],
                        "actions": [step["action"] for step in chunk],
                        "returns": returns,
                        "advantages": advantages,
                    }
                )

                # popart_returns.update(returns)
                # popart_advantages.update(advantages)

                # # Normalize returns and advantages
                # norm_returns = popart_returns.normalize(returns)
                # norm_advantages = popart_advantages.normalize(advantages)

                # D.append({
                #     "obs": [step["obs"] for step in chunk],
                #     "actions": [step["action"] for step in chunk],
                #     "returns": torch.tensor(norm_returns, dtype=torch.float32).to(device),
                #     "advantages": torch.tensor(norm_advantages, dtype=torch.float32).to(device)
                # })

            batch_rewards.append(episode_reward)
        episode_rewards.append(np.mean(batch_rewards))
        # Inner loop for minibatch updates with tqdm
        with tqdm(
            total=min(minibatch_count, len(D)), desc="Minibatch Updates", leave=False
        ) as mb_pbar:
            for _ in range(min(minibatch_count, len(D))):
                data = D[np.random.randint(len(D))]
                obs = torch.tensor(np.array(data["obs"]), dtype=torch.float32).to(
                    device
                )
                actions = torch.tensor(data["actions"], dtype=torch.int64).to(device)
                returns = data["returns"]
                advantages = data["advantages"]

                values = critic(obs).squeeze()

                action_probs = actor(obs)
                action_log_probs = torch.log(
                    action_probs.gather(1, actions.unsqueeze(1)).squeeze()
                )
                actor_loss = -(action_log_probs * advantages).mean()
                actor_loss = torch.clip(actor_loss, min=0)

                critic_loss = F.mse_loss(values, returns)
                critic_loss = torch.clip(critic_loss, min=0)

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                mb_pbar.update(1)  # Update the minibatch progress bar

        if step % 100 == 0:
            plot_losses_and_rewards(
                np.array(actor_losses),
                np.array(critic_losses),
                np.array(episode_rewards),
                iter=step,
                save=True,
            )

        step += 1
        pbar.update(1)  # Update the outer training progress bar

env.close()

torch.save(actor.state_dict(), "out/actor_weights.pth")
torch.save(critic.state_dict(), "out/critic_weights.pth")

actor_losses = np.array(actor_losses)
critic_losses = np.array(critic_losses)
episode_rewards = np.array(episode_rewards)

np.save("out/actor_losses.npy", actor_losses)
np.save("out/critic_losses.npy", critic_losses)
np.save("out/episode_rewards.npy", episode_rewards)

plot_losses_and_rewards(actor_losses, critic_losses, episode_rewards)
