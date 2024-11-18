from pettingzoo.butterfly import knights_archers_zombies_v10
from torch.optim import Adam
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    vector_state=True,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
    max_arrows=10
)
observations, infos = env.reset()


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    # def forward(self, obs, hidden):
        # rnn_out, hidden = self.rnn(obs.unsqueeze(1), hidden)
        # action_logits = self.fc(rnn_out.squeeze(1))
        # return action_logits, hidden
    
    def forward(self, obs, hidden):
        # Add sequence dimension if not present
        if obs.dim() == 2:  # Shape: (batch_size, input_size)
            obs = obs.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, input_size)
        rnn_out, hidden = self.rnn(obs, hidden)  # Correct input shape
        action_logits = self.fc(rnn_out.squeeze(1))  # Remove sequence dimension for logits
        return action_logits, hidden



class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    # def forward(self, state, hidden):
    #     rnn_out, hidden = self.rnn(state.unsqueeze(1), hidden)
    #     value = self.fc(rnn_out.squeeze(1))
    #     return value, hidden
    
    def forward(self, state, hidden):
        # Add sequence dimension if not present
        if state.dim() == 2:  # Shape: (batch_size, input_size)
            state = state.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, input_size)
        rnn_out, hidden = self.rnn(state, hidden)  # Correct input shape
        value = self.fc(rnn_out.squeeze(1))  # Remove sequence dimension for value
        return value, hidden


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages, [a + v for a, v in zip(advantages, values)]

# Initialize networks
obs_dim = env.observation_space("archer_0").shape[0]
action_dim = env.action_space("archer_0").n
state_dim = env.state().shape[0]
hidden_dim = 64

actor = Actor(obs_dim, action_dim, hidden_dim)
critic = Critic(state_dim, hidden_dim)

actor_optimizer = Adam(actor.parameters(), lr=3e-4)
critic_optimizer = Adam(critic.parameters(), lr=3e-4)

# Number of timesteps to train
max_timesteps = 1_000

# Initialize tracking variables
episode_rewards = []  # Stores cumulative reward for each episode
average_returns = []  # Smoothed average of rewards
sliding_window = 100  # Window size for smoothing
global_rewards = []   # Raw rewards for every timestep

from tqdm import tqdm
pbar = tqdm(total=max_timesteps, desc="Training Progress", unit="timesteps")

# Training loop (as discussed earlier)
global_timesteps = 0
observations, infos = env.reset()
current_episode_reward = 0
episode_count = 0

hidden_actor = {agent: torch.zeros(1, 1, hidden_dim) for agent in env.agents}
hidden_critic = torch.zeros(1, 1, hidden_dim)

while global_timesteps < max_timesteps:
    trajectory = []  # Collect trajectory for training
    episode_count += 1

    while True:  # Rollout loop within an episode
        actions = {}
        for agent, obs in observations.items():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, hidden_actor[agent] = actor(obs_tensor, hidden_actor[agent])
            probs = torch.softmax(logits, dim=-1)
            actions[agent] = torch.multinomial(probs, 1).item()

        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        trajectory.append((observations, actions, rewards, terminations, infos))
        global_timesteps += 1

        # Accumulate rewards for this episode
        current_episode_reward += sum(rewards.values())
        global_rewards.append(sum(rewards.values()))

        # Check for episode end
        if not env.agents:  # All agents are done
            observations, infos = env.reset()
            hidden_actor = {agent: torch.zeros(1, 1, hidden_dim) for agent in env.agents}
            hidden_critic = torch.zeros(1, 1, hidden_dim)
            break

        # Update observations
        observations = next_observations
        
    pbar.update(1)
    pbar.set_postfix({
        "Episode": episode_count
    })

    # Store episode reward and compute average return
    episode_rewards.append(current_episode_reward)
    current_episode_reward = 0  # Reset for the next episode

    if len(episode_rewards) >= sliding_window:
        average_return = sum(episode_rewards[-sliding_window:]) / sliding_window
    else:
        average_return = sum(episode_rewards) / len(episode_rewards)
    average_returns.append(average_return)

    # Optionally log progress
    if len(episode_rewards) % 10 == 0:
        print(f"Episode: {len(episode_rewards)}, Average Return: {average_return}")

pbar.close()

torch.save(actor.state_dict(), "actor.pth")
torch.save(critic.state_dict(), "critic.pth")
torch.save(average_returns, "average_returns.pth")

plt.figure(figsize=(10, 6))
plt.plot(range(len(average_returns)), average_returns, label="Average Return")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.title("Training Progress: Average Returns")
plt.legend()
plt.grid()
plt.show()
