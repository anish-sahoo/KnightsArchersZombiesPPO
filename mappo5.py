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

# from ReplayBuffer import ReplayBuffer 
# from ActorCritic import ActorCritic
from MetricsTracker import MetricsTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
    
    def push(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def get_batch(self):
        return self.buffer
    
    def clear(self):
        self.buffer = []

class ActorCritic(nn.Module):
    
    def __init__(self, action_dim):
        super().__init__()
        # Input: 84x84 grayscale
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)  # Output: 20x20x32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # Output: 9x9x64
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # Output: 7x7x64
        
        # Calculate flattened size: 7 * 7 * 64 = 3136
        self.fc = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        
        self.to(device)
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        # Add to ActorCritic __init__ after self.to(device):
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

# class MetricsTracker:
#     def __init__(self):
#         self.metrics = {
#             'episode_rewards': [],
#             'episode_lengths': [],
#             'policy_losses': [],
#             'value_losses': [],
#             'entropies': [],
#             'mean_values': [],
#             'mean_advantages': [],
#             'buffer_capacity_used': []
#         }
    
#     def add_metrics(self, **kwargs):
#         for key, value in kwargs.items():
#             if key in self.metrics:
#                 self.metrics[key].append(value)
    
#     def plot_metrics(self, save_dir='./metrics', save_individual=False):
#         import os
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
        
#         if save_individual:
#             for metric_name, values in self.metrics.items():
#                 if not values:
#                     continue
                    
#                 plt.figure(figsize=(10, 5))
#                 plt.plot(values)
#                 plt.title(f'{metric_name.replace("_", " ").title()}')
#                 plt.xlabel('Episode')
#                 plt.ylabel('Value')
#                 plt.savefig(f'{save_dir}/{metric_name}.png')
#                 plt.close()
            
#         # Plot combined training metrics
#         fig, axes = plt.subplots(3, 2, figsize=(15, 15))
#         fig.suptitle('Training Metrics')
        
#         axes[0, 0].plot(self.metrics['episode_rewards'])
#         axes[0, 0].set_title('Episode Rewards')
#         axes[0, 0].set_xlabel('Episode')
#         axes[0, 0].set_ylabel('Rewards')
        
#         axes[0, 1].plot(self.metrics['policy_losses'])
#         axes[0, 1].set_title('Policy Loss')
#         axes[0, 1].set_xlabel('Episode')
#         axes[0, 1].set_ylabel('Loss')
        
#         axes[1, 0].plot(self.metrics['value_losses'])
#         axes[1, 0].set_title('Value Loss')
#         axes[1, 0].set_xlabel('Episode')
#         axes[1, 0].set_ylabel('Loss')
        
#         axes[1, 1].plot(self.metrics['entropies'])
#         axes[1, 1].set_title('Entropy')
#         axes[1, 1].set_xlabel('Episode')
#         axes[1, 1].set_ylabel('Entropy')
        
#         axes[2, 0].plot(self.metrics['buffer_capacity_used'])
#         axes[2, 0].set_title('Buffer Capacity Used')
#         axes[2, 0].set_xlabel('Episode')
#         axes[2, 0].set_ylabel('Capacity')
        
#         # Hide the unused subplot
#         fig.delaxes(axes[2, 1])
        
#         plt.tight_layout()
#         plt.savefig(f'{save_dir}/combined_metrics.png')
#         plt.close()

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        
    advantages = torch.tensor(np.array(advantages), device=device)
    returns = advantages + torch.tensor(values, device=device)
    return advantages, returns

def ppo_update(model, optimizer, buffer, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4):
    batch = buffer.get_batch()
    
    # Convert states list to numpy array first for faster tensor conversion
    states = np.array([t['state'] for t in batch])
    # Add channel dimension if missing and convert to tensor
    if len(states.shape) == 3:  # If shape is (batch, height, width)
        states = states[:, None, :, :]  # Add channel dimension
    states = torch.FloatTensor(states).to(device)
    
    actions = torch.LongTensor([t['action'] for t in batch]).to(device)
    rewards = [t['reward'] for t in batch]
    dones = [t['done'] for t in batch]
    old_log_probs = torch.FloatTensor([t['log_prob'] for t in batch]).to(device)
    
    with torch.no_grad():
        _, values = model(states)
        values = values.squeeze()
    
    # Rest of the function remains the same
    advantages, returns = compute_gae(rewards, values.cpu().numpy(), dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(epochs):
        action_probs, new_values = model(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(new_values.squeeze(), returns)
        
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
    return policy_loss.item(), value_loss.item(), entropy.item()

def main():
    # Environment setup
    env = knights_archers_zombies_v10.parallel_env(
        render_mode=None,
        spawn_rate=15,
        vector_state=False,
        num_archers=2,
        num_knights=2,
        max_zombies=30,
    )
    
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize to 84x84
    env = ss.color_reduction_v0(env, mode='full')  # Grayscale
    
    # Training parameters
    max_episodes = 500
    max_steps = 500
    buffer = ReplayBuffer(max_size=3000)
    
    model = ActorCritic(env.action_space('archer_0').n)# , device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    episode_rewards = []
    metrics = MetricsTracker()
    
    reward_scale = 10
    penalty = 0.01
    
    for episode in tqdm(range(max_episodes)):
        observations, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            actions = {}
            step_transitions = []  # Temporary storage for this step's transitions
            
            # Get actions for all agents
            for agent in env.agents:
                obs = torch.FloatTensor(observations[agent]).to(device)
                with torch.no_grad():
                    action_probs, value = model(obs)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                actions[agent] = action.item()
                
                # Store transition in temporary list
                step_transitions.append({
                    'state': observations[agent],
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                })
            
            # Environment step
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_reward += sum(rewards.values())
            
            # Update transitions with rewards and dones and push to buffer
            for i, agent in enumerate(env.agents):
                step_transitions[i]['reward'] = rewards[agent] * reward_scale - penalty
                step_transitions[i]['done'] = terminations[agent] or truncations[agent]
                buffer.push(step_transitions[i])
            
            if all(terminations.values()) or all(truncations.values()):
                break
                
            observations = next_observations
        
        # PPO update
        if len(buffer.buffer) >= buffer.max_size:
            policy_loss, value_loss, entropy = ppo_update(model, optimizer, buffer, epochs=8)
            # Calculate mean values and advantages for the episode
            with torch.no_grad():
                # Convert list of states to numpy array first
                states_np = np.array([t['state'] for t in buffer.buffer])
                
                # Reshape to (batch_size, channels, height, width)
                if len(states_np.shape) == 3:  # If shape is (batch, height, width)
                    states_np = states_np[:, None, :, :]  # Add channel dimension
                
                # Convert to tensor
                states = torch.FloatTensor(states_np).to(device)
                
                _, values = model(states)
                mean_value = values.mean().item()
                advantages, _ = compute_gae([t['reward'] for t in buffer.buffer], 
                                        values.cpu().numpy(), 
                                        [t['done'] for t in buffer.buffer])
                mean_advantage = advantages.mean().item()

            metrics.add_metrics(
                episode_rewards=episode_reward,
                episode_lengths=step + 1,
                policy_losses=policy_loss,
                value_losses=value_loss,
                entropies=entropy,
                mean_values=mean_value,
                mean_advantages=mean_advantage
            )
            
            buffer.clear()
        
        episode_rewards.append(episode_reward)
    
        # Plotting
        if episode % 10 == 0:
            metrics.plot_metrics()
            metrics.plot_metrics_with_confidence_intervals()

if __name__ == "__main__":
    main()