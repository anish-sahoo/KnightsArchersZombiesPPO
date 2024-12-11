from pettingzoo.butterfly import knights_archers_zombies_v10
from torch.optim import Adam
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize the environment
env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    vector_state=True,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
)
observations, infos = env.reset()


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden):
        rnn_out, hidden = self.rnn(obs, hidden)
        action_logits = self.fc(rnn_out[:, -1])
        return action_logits, hidden


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, state, hidden):
        rnn_out, hidden = self.rnn(state, hidden)
        value = self.fc(rnn_out[:, -1])
        return value, hidden


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages, [a + v for a, v in zip(advantages, values)]


def train(timesteps=1_000_000):
    sequence_length, feature_dim = env.observation_space("archer_0").shape
    action_dim = env.action_space("archer_0").n
    hidden_dim = 64

    actor = Actor(input_dim=feature_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    critic = Critic(input_dim=feature_dim, hidden_dim=hidden_dim)

    actor_optimizer = Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = Adam(critic.parameters(), lr=3e-4)

    max_timesteps = timesteps
    # Training variables
    hidden_actor = {agent: torch.zeros(1, 1, hidden_dim) for agent in env.agents}
    hidden_critic = torch.zeros(1, 1, hidden_dim)

    # Tracking performance
    episode_rewards = []
    average_returns = []
    global_rewards = []
    sliding_window = 100

    # Training loop
    pbar = tqdm(total=max_timesteps, desc="Training Progress", unit="timesteps")
    global_timesteps = 0
    current_episode_reward = 0
    episode_count = 0

    while global_timesteps < max_timesteps:
        trajectory = []  # Collect trajectory for training
        observations, infos = env.reset()
        hidden_actor = {agent: torch.zeros(1, 1, hidden_dim) for agent in env.agents}
        hidden_critic = torch.zeros(1, 1, hidden_dim)

        while True:  # Rollout loop
            actions = {}
            for agent, obs in observations.items():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(
                    0
                )  # (batch_size=1, seq_len, input_size)
                logits, hidden_actor[agent] = actor(obs_tensor, hidden_actor[agent])
                probs = torch.softmax(logits, dim=-1)
                actions[agent] = torch.multinomial(probs, 1).item()

            # Step the environment
            next_observations, rewards, terminations, truncations, infos = env.step(
                actions
            )
            trajectory.append((observations, actions, rewards, terminations, infos))
            global_timesteps += 1
            current_episode_reward += sum(rewards.values())
            global_rewards.append(sum(rewards.values()))

            pbar.update(1)
            pbar.set_postfix({"Episodes": episode_count})

            # Check if all agents are done
            if not env.agents:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                break

            observations = next_observations

        episode_count += 1

        if len(episode_rewards) >= sliding_window:
            average_return = sum(episode_rewards[-sliding_window:]) / sliding_window
        else:
            average_return = sum(episode_rewards) / len(episode_rewards)
        average_returns.append(average_return)

        if timesteps % 100000 == 0:
            plt.plot(
                range(len(average_returns)), average_returns, label="Average Return"
            )
            plt.savefig(f"training_progress-{timesteps}.png")

    pbar.close()
    return actor, critic, average_returns


actor, critic, average_returns = train(500_000)

# Save models and results
torch.save(actor.state_dict(), "actor.pth")
torch.save(critic.state_dict(), "critic.pth")
torch.save(average_returns, "average_returns.pth")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(len(average_returns)), average_returns, label="Average Return")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.title("Training Progress: Average Returns")
plt.legend()
plt.grid()
plt.savefig("training_progress.png")
