import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import defaultdict
from pettingzoo.butterfly import knights_archers_zombies_v10
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        return F.softmax(self.network(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        return self.network(x)

class MAPPO:
    def __init__(self, env, hidden_dim=64, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
        # Reset environment to get initial observations and agent list
        observations, _ = self.env.reset()
        
        # Get dimensions from environment
        sample_agent = list(observations.keys())[0]
        
        # Flatten the observation space to get input dimension
        obs = observations[sample_agent]
        if isinstance(obs, np.ndarray):
            obs_dim = obs.size
        else:
            obs_dim = len(obs)
        
        act_dim = env.action_space(sample_agent).n
        print(f"Observation dimension: {obs_dim}")
        print(f"Action dimension: {act_dim}")
        
        # Create actor and critic networks for each agent type
        self.actors = {}
        self.critics = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        
        agent_types = ['knight', 'archer']
        for agent_type in agent_types:
            self.actors[agent_type] = ActorNetwork(obs_dim, hidden_dim, act_dim).to(device)
            self.critics[agent_type] = CriticNetwork(obs_dim, hidden_dim).to(device)
            self.actor_optimizers[agent_type] = optim.Adam(self.actors[agent_type].parameters(), lr=lr)
            self.critic_optimizers[agent_type] = optim.Adam(self.critics[agent_type].parameters(), lr=lr)

    def process_observation(self, obs):
        """Convert observation to flat tensor"""
        if isinstance(obs, np.ndarray):
            flat_obs = obs.flatten()
        else:
            flat_obs = np.array(obs)
        return torch.FloatTensor(flat_obs)

    def get_action(self, obs, agent):
        agent_type = 'knight' if 'knight' in agent else 'archer'
        actor = self.actors[agent_type]
        
        # Process observation
        obs_tensor = self.process_observation(obs).to(device)
        
        with torch.no_grad():
            probs = actor(obs_tensor)
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.cpu().item()

    def train_step(self, trajectories):
        for agent_type in self.actors.keys():
            # Filter trajectories for current agent type
            agent_trajectories = [t for t in trajectories if (agent_type in t['agent'])]
            
            if not agent_trajectories:
                continue
                
            # Prepare batch data
            states = torch.stack([self.process_observation(t['state']) for t in agent_trajectories]).to(device)
            actions = torch.LongTensor([t['action'] for t in agent_trajectories]).to(device)
            rewards = torch.FloatTensor([t['reward'] for t in agent_trajectories]).to(device)
            next_states = torch.stack([self.process_observation(t['next_state']) for t in agent_trajectories]).to(device)
            old_log_probs = torch.FloatTensor([t['log_prob'] for t in agent_trajectories]).to(device)
            
            # Compute advantages
            with torch.no_grad():
                values = self.critics[agent_type](states).squeeze()
                next_values = self.critics[agent_type](next_states).squeeze()
                advantages = rewards + self.gamma * next_values - values
            
            # Update critic
            value_loss = F.mse_loss(self.critics[agent_type](states).squeeze(), rewards + self.gamma * next_values)
            self.critic_optimizers[agent_type].zero_grad()
            value_loss.backward()
            self.critic_optimizers[agent_type].step()
            
            # Update actor
            probs = self.actors[agent_type](states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizers[agent_type].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_type].step()
            
            return value_loss, actor_loss

    def train(self, num_episodes=1000, max_steps=1000):
        rewards_per_episode = []
        knight_rewards = []
        archer_rewards = []
        actor_losses = []
        critic_losses = []
        
        for _ in tqdm(range(num_episodes)):
            observations, _ = self.env.reset()  # Reset environment to start new episode
            episode_trajectories = []
            episode_rewards = defaultdict(list)
            
            for step in range(max_steps):
                actions = {}
                action_log_probs = {}
                
                # Get actions for all agents simultaneously
                for agent, obs in observations.items():
                    action, log_prob = self.get_action(obs, agent)
                    actions[agent] = action
                    action_log_probs[agent] = log_prob
                
                # Take a single environment step for all agents at once
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Store trajectories and rewards
                for agent, obs in observations.items():
                    if agent in next_observations:
                        trajectory = {
                            'agent': agent,
                            'state': obs,
                            'action': actions[agent],
                            'reward': rewards[agent],
                            'next_state': next_observations[agent],
                            'log_prob': action_log_probs[agent]
                        }
                        episode_trajectories.append(trajectory)
                        episode_rewards[agent].append(rewards[agent])
                
                # Update observations for next step
                observations = next_observations
                
                # Check if all agents are done (either terminated or truncated)
                if all(terminations.values()) or all(truncations.values()):
                    break
            
            value_loss, actor_loss = self.train_step(episode_trajectories)
            actor_losses.append(actor_loss.item())
            critic_losses.append(value_loss.item())
            
            total_avg_reward = np.mean([reward for agent_rewards in episode_rewards.values() for reward in agent_rewards])
            knight_rewards.append(np.mean([reward for agent, rewards in episode_rewards.items() if 'knight' in agent for reward in rewards]))
            archer_rewards.append(np.mean([reward for agent, rewards in episode_rewards.items() if 'archer' in agent for reward in rewards]))
            rewards_per_episode.append(total_avg_reward)
        return rewards_per_episode, knight_rewards, archer_rewards, actor_losses, critic_losses


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    env = knights_archers_zombies_v10.parallel_env(
        num_archers=2,
        num_knights=2,
        max_zombies=50,
        killable_knights=True,
        killable_archers=True,
        line_death=True,
    )
    
    mappo = MAPPO(env)
    rewards, knight_r, archer_r, actor_l, critic_l = mappo.train(num_episodes=500, max_steps=1000)
    plt.plot(rewards, label="Total Average Reward")
    plt.plot(knight_r, label="Knight Average Reward")
    plt.plot(archer_r, label="Archer Average Reward")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("MAPPO Training")
    plt.savefig("mappo_rewards_1000ep_0001.png")
    plt.show()
    
    plt.figure()
    plt.plot(actor_l, label="Actor Loss")
    plt.plot(critic_l, label="Critic Loss")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("MAPPO Actor and Critic Losses")
    plt.savefig("mappo_losses_1000ep_0001.png")
    plt.show()
    
    for agent_type in mappo.actors.keys():
        torch.save(mappo.actors[agent_type].state_dict(), f"{agent_type}_actor.pth")
        torch.save(mappo.critics[agent_type].state_dict(), f"{agent_type}_critic.pth")
    env.close()

if __name__ == "__main__":
    main()