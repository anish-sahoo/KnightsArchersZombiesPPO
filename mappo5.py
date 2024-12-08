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
        self.conv1 = nn.Conv2d(4, 16, 3, stride=2)  # Output: 41x41x16
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)  # Output: 20x20x32
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)  # Output: 9x9x32

        # Flattened size: 32 * 9 * 9 = 2592
        self.fc = nn.Linear(2592, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        
        # # Input: 84x84 grayscale
        # self.conv1 = nn.Conv2d(1, 32, 8, stride=4)  # Output: 20x20x32
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # Output: 9x9x64
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # Output: 7x7x64
        
        # # Calculate flattened size: 7 * 7 * 64 = 3136
        # self.fc = nn.Linear(3136, 256)
        # self.actor = nn.Linear(256, action_dim)
        # self.critic = nn.Linear(256, 1)
        
        self.to(device)
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        # Add to ActorCritic __init__ after self.to(device):
        self.apply(init_weights)

    # def forward(self, x):
    #     # Handle observation preprocessing
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.FloatTensor(x)
        
    #     # Move to device if not already there
    #     if x.device != device:
    #         x = x.to(device)
        
    #     # Ensure proper dimensions
    #     if len(x.shape) == 2:  # If input is (84, 84)
    #         x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims -> (1, 1, 84, 84)
    #     elif len(x.shape) == 3:  # If input is (1, 84, 84) or (3, 84, 84)
    #         if x.shape[0] == 3:  # If RGB
    #             x = x.mean(dim=0, keepdim=True).unsqueeze(0)  # Convert to grayscale and add batch
    #         else:
    #             x = x.unsqueeze(0)  # Just add batch dimension
        
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, 3136)
    #     x = F.relu(self.fc(x))
        
    #     action_probs = F.softmax(self.actor(x), dim=-1)
    #     value = self.critic(x)
    #     return action_probs, value
    
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
            x = x.permute(0, 3, 1, 2)  # Reorder to (batch_size, channels, height, width)
        
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

def ppo_update(model, optimizer, buffer, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.02, epochs=4, minibatch_size=32):
    batch = buffer.get_batch()
    
    # np.random.shuffle(batch)
    
    # batch = batch[:minibatch_size]  # Mini-batch instead of whole buffer ---------------------------this is new
    
    indices = np.random.choice(len(batch), minibatch_size, replace=False)
    batch = [batch[i] for i in indices]
    
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
        
        # value_loss = F.mse_loss(new_values.squeeze(), returns)
        value_loss = F.huber_loss(new_values.squeeze(), returns, delta=10)
        
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
    return policy_loss.item(), value_loss.item(), entropy.item()

def main(hyperparameters, name=''):
    # Environment setup
    env = knights_archers_zombies_v10.parallel_env(
        render_mode=None,
        spawn_rate=15,
        vector_state=False,
        num_archers=2, #2
        num_knights=2, #2
        max_zombies=30,
    )
    
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize to 84x84
    env = ss.color_reduction_v0(env, mode='full')  # Grayscale
    env = ss.frame_stack_v2(env, stack_size=4) # Stack 4 frames
    
    max_timesteps = hyperparameters['max_timesteps']
    max_steps = hyperparameters['max_steps']
    buffer = ReplayBuffer(max_size=hyperparameters['buffer_size'])
    lr = hyperparameters['lr']
    reward_scale = hyperparameters['reward_scale']
    penalty = hyperparameters['penalty']
    
    # ppo update hyperparameters
    epochs = hyperparameters['epochs']
    minibatch_size = hyperparameters['minibatch_size']
    clip_epsilon = hyperparameters['clip_epsilon']
    value_coef = hyperparameters['value_coef']
    entropy_coef = hyperparameters['entropy_coef']
    
    # gae hyperparameters
    gamma = hyperparameters['gamma']
    gae_lambda = hyperparameters['gae_lambda']
    
    writer = SummaryWriter(log_dir=f'./runs/mappo_{name}')
    
    model = ActorCritic(env.action_space('archer_0').n)# , device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    episode_rewards = []
    metrics = MetricsTracker(name)
    
    # plot episode numbers will be much smaller as it takes multiple timesteps to complete an episode (max out the replay buffer and then do an update)

    for timestep in tqdm(range(max_timesteps), desc='Training', unit=' step'):
        observations, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            actions = {}
            trajectory = []  # Temporary storage for this step's transitions
            
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
                trajectory.append({
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
                trajectory[i]['reward'] = rewards[agent] * reward_scale - penalty
                trajectory[i]['done'] = terminations[agent] or truncations[agent]
                buffer.push(trajectory[i])
            
            if all(terminations.values()) or all(truncations.values()):
                break
                
            observations = next_observations
        
        # PPO update
        if len(buffer.buffer) > minibatch_size: # buffer.max_size:
            policy_loss, value_loss, entropy = ppo_update(
                model, 
                optimizer, 
                buffer, 
                epochs=epochs, 
                minibatch_size=minibatch_size, 
                clip_epsilon=clip_epsilon, 
                value_coef=value_coef, 
                entropy_coef=entropy_coef
            )
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
                                        [t['done'] for t in buffer.buffer],
                                        gamma=gamma,
                                        gae_lambda=gae_lambda
                                )
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
            
            writer.add_scalar('Loss/Policy', policy_loss, timestep)
            writer.add_scalar('Loss/Value', value_loss, timestep)
            writer.add_scalar('Loss/Entropy', entropy, timestep)
            writer.add_scalar('Values/MeanValue', mean_value, timestep)
            writer.add_scalar('Values/MeanAdvantage', mean_advantage, timestep)
            writer.add_scalar('Rewards/Episode', episode_reward, timestep)
            writer.add_scalar('EpisodeLength', step + 1, timestep)
            
            buffer.clear()
        
        episode_rewards.append(episode_reward)
    
        # Plotting
        if timestep % 10 == 0 and timestep > 0:
            metrics.plot_metrics(save_individual=True)
            metrics.plot_metrics_smooth(save_individual=True)
            metrics.save_metrics()
            
        if timestep % 500 == 0 and timestep > 0:
            savedir = f'./models/{name}' if name != '' else './models'
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            
            filename = f'{savedir}/mappo_model_episode_{timestep}_{time.time()}.pth'
            if name != '':
                filename = f'{savedir}/mappo_model_episode_{timestep}_{time.time()}.pth'
            torch.save(model.state_dict(), filename)
            metrics.save_metrics()
            
        # if timestep % 50 == 0:
        #     # Evaluate policy performance
        #     eval_rewards = []
        #     eval_steps = []
        #     eval_episodes = 5

        #     for _ in range(eval_episodes):
        #         eval_observations, _ = env.reset()
        #         eval_episode_reward = 0
        #         eval_episode_steps = 0

        #         while True:
        #             eval_actions = {}
        #             for agent in env.agents:
        #                 eval_obs = torch.FloatTensor(eval_observations[agent]).to(device)
        #                 with torch.no_grad():
        #                     eval_action_probs, _ = model(eval_obs)
        #                     eval_dist = Categorical(eval_action_probs)
        #                     eval_action = eval_dist.sample()
        #                 eval_actions[agent] = eval_action.item()

        #             eval_next_observations, eval_rewards_dict, eval_terminations, eval_truncations, _ = env.step(eval_actions)
        #             eval_episode_reward += sum(eval_rewards_dict.values())
        #             eval_episode_steps += 1

        #             if all(eval_terminations.values()) or all(eval_truncations.values()):
        #                 break

        #             eval_observations = eval_next_observations

        #         eval_rewards.append(eval_episode_reward)
        #         eval_steps.append(eval_episode_steps)

        #     avg_eval_reward = np.mean(eval_rewards)
        #     avg_eval_steps = np.mean(eval_steps)

        #     writer.add_scalar('Evaluation/AverageReward', avg_eval_reward, timestep)
        #     writer.add_scalar('Evaluation/AverageSteps', avg_eval_steps, timestep)

    env.close()
    writer.close()
    
    savedir = f'./models/{name}' if name != '' else './models'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    filename = f'{savedir}/mappo_model_episode_{timestep}_{time.time()}.pth'
    if name != '':
        filename = f'{savedir}/mappo_model_episode_{timestep}_{time.time()}.pth'
    torch.save(model.state_dict(), filename)
    metrics.save_metrics()
    
if __name__ == "__main__":
    hyperparameters = {
        'max_timesteps': 10000,
        'max_steps': 5000,
        'buffer_size': 8000,
        'lr': 5e-5,
        'reward_scale': 1,
        'penalty': 0,
        'epochs': 10,
        'minibatch_size': 256,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.03, #0.02,
        'gamma': 0.99,
        'gae_lambda': 0.95
    }
    main(hyperparameters=hyperparameters, name='run12_only_archers')
    
# first run hyperparameters
# max_timesteps = 500
# max_steps = 500
# buffer = ReplayBuffer(max_size=3000)
# lr = 0.001 #3e-4
# reward_scale = 10
# penalty = 0.01
# epochs = 8
# model = ActorCritic(env.action_space('archer_0').n)# , device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# ppo_update(model, optimizer, buffer, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4)
# compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)