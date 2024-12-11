import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10
from torch.optim import Adam
from torch.distributions import Categorical
from tqdm import tqdm


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


def compute_ppo_loss(advantages, ratio, clip_param=0.2):
    return -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages,
    ).mean()


def prepare_batch(trajectory, sequence_length, critic):
    """
    Prepare batch of trajectories for training, including advantage estimation
    """
    # Get the first trajectory step to identify agents
    first_step = trajectory[0]
    first_observations = first_step[0]  # observations dict
    agents = list(first_observations.keys())

    # Initialize dictionaries for each agent
    agent_observations = {agent: [] for agent in agents}
    agent_actions = {agent: [] for agent in agents}
    agent_old_log_probs = {agent: [] for agent in agents}
    agent_rewards = {agent: [] for agent in agents}
    agent_dones = {agent: [] for agent in agents}

    # Organize data by agent
    for t in range(len(trajectory)):
        obs_dict, actions_dict, old_log_probs_dict, rewards_dict, done = trajectory[t]

        for agent in agents:
            agent_observations[agent].append(obs_dict[agent])
            agent_actions[agent].append(actions_dict[agent])
            agent_old_log_probs[agent].append(old_log_probs_dict[agent])
            agent_rewards[agent].append(rewards_dict[agent])
            agent_dones[agent].append(done if isinstance(done, bool) else done[agent])

    # Convert to numpy arrays
    for agent in agents:
        agent_observations[agent] = np.array(agent_observations[agent])
        agent_actions[agent] = np.array(agent_actions[agent])
        agent_old_log_probs[agent] = np.array(agent_old_log_probs[agent])
        agent_rewards[agent] = np.array(agent_rewards[agent])
        agent_dones[agent] = np.array(agent_dones[agent])

    # Compute advantages and returns for each agent
    agent_advantages = {}
    agent_returns = {}

    for agent in agents:
        # Get values for the entire trajectory
        obs_tensor = torch.tensor(agent_observations[agent], dtype=torch.float32)
        with torch.no_grad():
            values, _ = critic(obs_tensor, torch.zeros(1, 1, critic.rnn.hidden_size))
            values = values.squeeze().numpy()

        # Compute next values (shift values one step forward and append 0)
        next_values = np.append(values[1:], 0)

        # Compute GAE advantages
        advantages = []
        gae = 0
        gamma = 0.99
        lam = 0.95

        for t in reversed(range(len(agent_rewards[agent]))):
            delta = (
                agent_rewards[agent][t]
                + gamma * next_values[t] * (1 - agent_dones[agent][t])
                - values[t]
            )
            gae = delta + gamma * lam * (1 - agent_dones[agent][t]) * gae
            advantages.insert(0, gae)

        agent_advantages[agent] = np.array(advantages)
        agent_returns[agent] = agent_advantages[agent] + values

    # Split into chunks of sequence_length
    num_chunks = len(trajectory) // sequence_length
    if num_chunks == 0:  # Handle case where trajectory is shorter than sequence_length
        num_chunks = 1
        actual_sequence_length = len(trajectory)
    else:
        actual_sequence_length = sequence_length

    batches = []
    for i in range(num_chunks):
        start_idx = i * actual_sequence_length
        end_idx = start_idx + actual_sequence_length

        batch = {
            "observations": {
                agent: agent_observations[agent][start_idx:end_idx] for agent in agents
            },
            "actions": {
                agent: agent_actions[agent][start_idx:end_idx] for agent in agents
            },
            "old_log_probs": {
                agent: agent_old_log_probs[agent][start_idx:end_idx] for agent in agents
            },
            "returns": {
                agent: agent_returns[agent][start_idx:end_idx] for agent in agents
            },
            "advantages": {
                agent: agent_advantages[agent][start_idx:end_idx] for agent in agents
            },
            "initial_hidden_states_actor": {
                agent: torch.zeros(1, 1, critic.rnn.hidden_size) for agent in agents
            },
            "initial_hidden_states_critic": {
                agent: torch.zeros(1, 1, critic.rnn.hidden_size) for agent in agents
            },
        }
        batches.append(batch)

    return batches


def update_network(
    batch,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    hidden_dim,
    device="cuda",
    clip_param=0.2,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=0.5,
):
    """
    Update actor and critic networks using PPO algorithm with RNN support
    """
    # Initialize hidden states
    hidden_actor = {
        agent: torch.zeros(1, 1, hidden_dim).to(device)
        for agent in batch["observations"].keys()
    }
    hidden_critic = {
        agent: torch.zeros(1, 1, hidden_dim).to(device)
        for agent in batch["observations"].keys()
    }

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0

    # Process each agent's data
    for agent_id in batch["observations"].keys():
        # Get agent-specific data
        obs = torch.tensor(batch["observations"][agent_id], dtype=torch.float32).to(
            device
        )
        actions = torch.tensor(batch["actions"][agent_id], dtype=torch.long).to(device)
        old_log_probs = torch.tensor(
            batch["old_log_probs"][agent_id], dtype=torch.float32
        ).to(device)
        advantages = torch.tensor(
            batch["advantages"][agent_id], dtype=torch.float32
        ).to(device)
        returns = torch.tensor(batch["returns"][agent_id], dtype=torch.float32).to(
            device
        )

        # Forward pass through actor
        logits, hidden_actor[agent_id] = actor(obs.unsqueeze(0), hidden_actor[agent_id])
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Compute log probabilities and ratio
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Compute policy loss
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages,
        ).mean()

        # Compute entropy bonus
        entropy = dist.entropy().mean()

        # Forward pass through critic
        value, hidden_critic[agent_id] = critic(
            obs.unsqueeze(0), hidden_critic[agent_id]
        )

        # Compute value loss
        value_loss = F.mse_loss(value.squeeze(), returns)

        # Combine losses
        total_loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

        # Update networks
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

        actor_optimizer.step()
        critic_optimizer.step()

        # Accumulate losses for logging
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()

    # Average losses across agents
    num_agents = len(batch["observations"].keys())
    return {
        "policy_loss": total_policy_loss / num_agents,
        "value_loss": total_value_loss / num_agents,
        "entropy": total_entropy / num_agents,
    }


def train_recurrent_mappo(
    env, max_timesteps, sequence_length, hidden_dim, device="cuda"
):
    # Training variables
    global_timesteps = 0
    episode_count = 0
    episode_rewards = []
    average_returns = []
    sliding_window = 100

    sequence_length, feature_dim = env.observation_space("archer_0").shape
    action_dim = env.action_space("archer_0").n

    actor = Actor(input_dim=feature_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    critic = Critic(input_dim=feature_dim, hidden_dim=hidden_dim)

    actor_optimizer = Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = Adam(critic.parameters(), lr=3e-4)

    pbar = tqdm(total=max_timesteps, desc="Training Progress", unit="timesteps")

    while global_timesteps < max_timesteps:
        observations, infos = env.reset()
        hidden_actor = {
            agent: torch.zeros(1, 1, hidden_dim).to(device) for agent in env.agents
        }

        trajectory = []
        episode_reward = 0

        # Collect trajectory
        while True:
            actions = {}
            old_log_probs = {}

            # Get actions for each agent
            for agent_id, obs in observations.items():
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                )
                logits, hidden_actor[agent_id] = actor(
                    obs_tensor, hidden_actor[agent_id]
                )
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                actions[agent_id] = action.item()
                old_log_probs[agent_id] = dist.log_prob(action).item()

            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(
                actions
            )
            episode_reward += sum(rewards.values())

            # Increment global timesteps
            global_timesteps += 1
            pbar.update(1)

            # Store transition
            trajectory.append(
                (
                    observations,
                    actions,
                    old_log_probs,
                    rewards,
                    terminations or truncations,
                )
            )

            if (
                not env.agents
                or any(terminations.values())
                or any(truncations.values())
            ):
                break

            observations = next_observations

            # Check if we've reached max timesteps
            if global_timesteps >= max_timesteps:
                break

        # Episode completed
        episode_count += 1
        episode_rewards.append(episode_reward)

        # Calculate average return
        if len(episode_rewards) >= sliding_window:
            average_return = sum(episode_rewards[-sliding_window:]) / sliding_window
        else:
            average_return = sum(episode_rewards) / len(episode_rewards)
        average_returns.append(average_return)

        # Process trajectory
        batches = prepare_batch(trajectory, sequence_length, critic)

        # Update networks
        for batch in batches:
            losses = update_network(
                batch,
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                hidden_dim,
                device,
            )

        # Logging
        if episode_count % 10 == 0:
            pbar.set_postfix(
                {
                    "Episodes": episode_count,
                    "Avg Return": f"{average_return:.2f}",
                    "Episode Reward": f"{episode_reward:.2f}",
                }
            )

        # Optional: Save progress plots periodically
        if global_timesteps % 100000 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(len(average_returns)), average_returns, label="Average Return"
            )
            plt.xlabel("Episode")
            plt.ylabel("Average Return")
            plt.title("Training Progress: Average Returns")
            plt.legend()
            plt.grid()
            plt.savefig(f"training_progress_{max_timesteps}.png")
            plt.close()

    pbar.close()
    return actor, critic, average_returns


env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    vector_state=True,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
)
observations, infos = env.reset()

actor, critic, rewards = train_recurrent_mappo(
    env=env, max_timesteps=50_000, sequence_length=20, hidden_dim=64, device="cpu"
)
