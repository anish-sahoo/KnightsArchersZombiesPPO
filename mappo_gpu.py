import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pettingzoo.butterfly import knights_archers_zombies_v10
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# device = torch.device("cpu")

# PopArt normalization (unchanged)
class PopArt:
    def __init__(self, beta=0.0003, epsilon=1e-5):
        self.mean = 0.0
        self.sq_mean = 0.0
        self.beta = beta
        self.epsilon = epsilon

    def normalize(self, x):        
        self.mean = self.beta * np.mean(x) + (1 - self.beta) * self.mean
        self.sq_mean = self.beta * np.mean(x ** 2) + (1 - self.beta) * self.sq_mean
        std = np.sqrt(self.sq_mean - self.mean ** 2 + self.epsilon)
        return (x - self.mean) / std

# Actor and Critic networks with CNN moved to device
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Output: (32, 127, 127)
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 62, 62)
            nn.ReLU(inplace=False),
            nn.Flatten()
        )
        conv_output_size = 64 * 62 * 62
        self.rnn = nn.GRU(conv_output_size, 256, batch_first=True)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        self.apply(self._init_weights)
    
    def forward(self, x, h):
        # Ensure input is on the correct device
        x = x.to(device)
        if h is not None:
            h = h.to(device)
        
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape without using view to avoid potential in-place issues
        x = x.reshape(batch_size * seq_len, C, H, W)
        x = self.conv(x)
        x = x.reshape(batch_size, seq_len, -1)
        
        # Ensure h is initialized correctly if None
        if h is None:
            h = torch.zeros(1, batch_size, 256, device=device)
        
        x, h = self.rnn(x, h)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value, h

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

# Environment setup
env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=20,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=30,
)

env.reset()
agent_names = env.agents
observation_space = env.observation_space(agent_names[0])
action_space = env.action_space(agent_names[0])

input_channels = observation_space.shape[2]  # Should be 3 for RGB images
action_dim = action_space.n

# Initialize networks and optimizer
policy_net = ActorCritic(input_channels, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
popart = PopArt()

# Initialize lists to store losses and rewards
all_rewards = [] # Store the average reward for each training step
all_losses = [] # Store the loss for each training step
total_steps = 10 # Total number of training steps
batch_size = 1 # number of episodes to collect before updating the networks
T = 4 # number of trajectories
chunk_size = 20 # L

# Training loop
step = 0
with tqdm(total=total_steps, desc="Training") as pbar:
    while step <= total_steps:
        D = []
        episode_rewards = []
        with tqdm(total=batch_size, desc="Collecting Trajectories", leave=False) as collect_pbar:
            for _ in range(batch_size):  # batch_size = 4
                trajectories = {agent: [] for agent in agent_names}
                h_policies = {agent: None for agent in agent_names}
                obs, _ = env.reset()
                done = {agent: False for agent in env.agents}
                total_reward = 0
                for t in range(T):  # T = 100
                    actions = {}
                    log_probs = {}
                    values = {}
                    for agent in env.agents:
                        if not done[agent]:
                            obs_array = np.transpose(obs[agent], (2, 0, 1))  # Convert to (C, H, W)
                            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                            if h_policies[agent] is None:
                                h_policies[agent] = torch.zeros(1, 1, 256, device=device)
                            else:
                                h_policies[agent] = h_policies[agent].detach()
                            
                            policy_logits, value, h_policy = policy_net(obs_tensor, h_policies[agent])
                            dist = Categorical(logits=policy_logits.squeeze(1))
                            action = dist.sample()
                            actions[agent] = action.item()
                            log_probs[agent] = dist.log_prob(action)
                            values[agent] = value.squeeze(1)
                            h_policies[agent] = h_policy.detach()
                        else:
                            actions[agent] = None
                    next_obs, rewards, terminations, truncations, _ = env.step(actions)
                    for agent in env.agents:
                        if not done[agent]:
                            trajectories[agent].append({
                                'obs': obs[agent],
                                'action': actions[agent],
                                'log_prob': log_probs[agent],
                                'value': values[agent],
                                'reward': rewards[agent],
                                'done': terminations[agent] or truncations[agent]
                            })
                            done[agent] = terminations[agent] or truncations[agent]
                            total_reward = total_reward + rewards[agent]
                    obs = next_obs
                    if all(done.values()):
                        break
                episode_rewards.append(total_reward)
                D.append(trajectories)
                collect_pbar.update(1)
        all_rewards.append(np.mean(episode_rewards))

        # Compute advantages and returns using GAE and normalize with PopArt
        for trajectories in D:
            for agent in trajectories:
                traj = trajectories[agent]
                rewards = [step['reward'] for step in traj]
                values = [step['value'].item() for step in traj] + [0]
                dones = [step['done'] for step in traj]
                advantages = []
                gae = 0
                for i in reversed(range(len(traj))):
                    mask = 1.0 - dones[i]
                    delta = rewards[i] + 0.99 * values[i + 1] * mask - values[i]
                    gae = delta + 0.99 * 0.95 * mask * gae
                    advantages.insert(0, gae)
                returns = [adv + val for adv, val in zip(advantages, values[:-1])]
                # Normalize returns using PopArt
                returns_np = np.array(returns)
                normalized_returns = popart.normalize(returns_np)
                # Store advantages and normalized returns in trajectory
                for idx, step_data in enumerate(traj):
                    step_data['advantage'] = advantages[idx]
                    step_data['return'] = normalized_returns[idx]

        # Prepare batches and update networks
        # Split trajectories into chunks of length 20 (L = 20)
        
        all_agent_data = []
        for trajectories in D:
            for agent in trajectories:
                traj = trajectories[agent]
                for l in range(0, len(traj), chunk_size):
                    chunk = traj[l:l+chunk_size]
                    all_agent_data.append(chunk)

        # Shuffle and create minibatches
        random.shuffle(all_agent_data)
        num_mini_batches = max(1, len(all_agent_data) // 4)  # batch_size = 4
        total_loss = 0
        with tqdm(total=num_mini_batches, desc="Updating Networks", leave=False) as update_pbar:
            for _ in range(num_mini_batches):
                minibatch = [all_agent_data.pop() for _ in range(4) if all_agent_data]
                # Prepare batch data
                obs_batch = []
                actions_batch = []
                advantages_batch = []
                returns_batch = []
                old_log_probs_batch = []
                for chunk in minibatch:
                    obs_chunk = np.stack([np.transpose(step['obs'], (2, 0, 1)) for step in chunk])  # Shape: (seq_len, C, H, W)
                    obs_tensor = torch.from_numpy(obs_chunk).float()
                    obs_batch.append(obs_tensor)
                    actions_batch.extend([step['action'] for step in chunk])
                    advantages_batch.extend([step['advantage'] for step in chunk])
                    returns_batch.extend([step['return'] for step in chunk])
                    old_log_probs_batch.extend([step['log_prob'] for step in chunk])

                # Stack and move batch data to device
                obs_batch = torch.stack(obs_batch).to(device)  # Shape: [batch_size, seq_len, C, H, W]
                actions_batch = torch.tensor(actions_batch, device=device)
                advantages_batch = torch.tensor(advantages_batch, dtype=torch.float32, device=device)
                returns_batch = torch.tensor(returns_batch, dtype=torch.float32, device=device)
                old_log_probs_batch = torch.stack(old_log_probs_batch).to(device)

                # Reshape tensors to match expected shapes
                batch_size, seq_len = obs_batch.shape[0], obs_batch.shape[1]
                actions_batch = actions_batch.reshape(batch_size, seq_len)
                advantages_batch = advantages_batch.reshape(batch_size, seq_len)
                returns_batch = returns_batch.reshape(batch_size, seq_len)
                old_log_probs_batch = old_log_probs_batch.reshape(batch_size, seq_len)

                # Forward pass
                policy_logits, value_preds, _ = policy_net(obs_batch, None)  # Shapes: [batch_size, seq_len, action_dim] and [batch_size, seq_len, 1]
                value_preds = value_preds.squeeze(-1)  # Shape: [batch_size, seq_len]
                dist = Categorical(logits=policy_logits)

                # Compute new log probabilities
                new_log_probs = dist.log_prob(actions_batch)  # Shape: [batch_size, seq_len]

                # Reshape tensors for loss computation
                new_log_probs = new_log_probs.reshape(-1)    # Shape: [batch_size * seq_len]
                advantages_batch = advantages_batch.reshape(-1)
                returns_batch = returns_batch.reshape(-1)
                value_preds = value_preds.reshape(-1)
                old_log_probs_batch = old_log_probs_batch.reshape(-1)

                # Compute ratio and surrogate losses
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.MSELoss()(value_preds, returns_batch)

                # Compute entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                optimizer.step()

                total_loss = total_loss + loss.item()
                update_pbar.update(1)
        all_losses.append(total_loss / num_mini_batches)
        step = step + 1
        pbar.update(1)

# Plot the losses and rewards
plt.figure()
plt.plot(all_losses)
plt.title('Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(all_rewards)
plt.title('Average Episode Reward')
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.show()

env.close()