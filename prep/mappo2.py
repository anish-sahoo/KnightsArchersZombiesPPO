import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gymnasium as gym
from pettingzoo.butterfly import knights_archers_zombies_v10
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class MultiAgentPPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon_clip=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        
        # Initialize networks and optimizers for each agent
        self.agents = {}
        self.optimizers = {}
        
        # Get the first observation to determine observation space
        self.env.reset()
        for agent in self.env.agents:
            obs_space = len(self.env.observation_space(agent).low)
            action_space = self.env.action_space(agent).n
            
            self.agents[agent] = ActorCritic(obs_space, action_space)
            self.optimizers[agent] = optim.Adam(self.agents[agent].parameters(), lr=learning_rate)
    
    def get_action(self, state, agent):
        state = torch.FloatTensor(state)
        action_probs, _ = self.agents[agent](state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        gae = 0
        next_value = next_value.detach()
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_return = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_return = returns[0]
                
            delta = rewards[step] + self.gamma * next_return * next_non_terminal - values[step]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def update(self, states, actions, log_probs, returns, advantages, agent):
        for _ in range(self.epochs):
            current_action_probs, current_values = self.agents[agent](states)
            dist = Categorical(current_action_probs)
            current_log_probs = dist.log_prob(actions)
            
            ratios = torch.exp(current_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_values, returns)
            entropy_loss = -0.01 * dist.entropy().mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            self.optimizers[agent].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent].parameters(), 0.5)
            self.optimizers[agent].step()
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            episode_rewards = {agent: [] for agent in self.env.agents}
            states = {agent: [] for agent in self.env.agents}
            actions = {agent: [] for agent in self.env.agents}
            log_probs = {agent: [] for agent in self.env.agents}
            values = {agent: [] for agent in self.env.agents}
            dones = {agent: [] for agent in self.env.agents}
            
            observations, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                for agent in self.env.agents:
                    state = observations[agent]
                    action, log_prob = self.get_action(state, agent)
                    
                    with torch.no_grad():
                        _, value = self.agents[agent](torch.FloatTensor(state))
                    
                    states[agent].append(state)
                    actions[agent].append(action)
                    log_probs[agent].append(log_prob)
                    values[agent].append(value)
                    
                    next_observations, rewards, terminations, truncations, _ = self.env.step(action)
                    done = terminations[agent] or truncations[agent]
                    
                    episode_rewards[agent].append(rewards[agent])
                    dones[agent].append(done)
                    
                    if done:
                        break
                        
                    observations = next_observations
            
            # Update policy for each agent
            for agent in self.env.agents:
                with torch.no_grad():
                    _, next_value = self.agents[agent](torch.FloatTensor(observations[agent]))
                
                returns = self.compute_returns(
                    episode_rewards[agent],
                    dones[agent],
                    torch.cat(values[agent]).detach(),
                    next_value
                )
                
                returns = torch.tensor(returns)
                advantages = returns - torch.cat(values[agent]).detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                self.update(
                    torch.FloatTensor(np.array(states[agent])),
                    torch.tensor(actions[agent]),
                    torch.cat(log_probs[agent]),
                    returns,
                    advantages,
                    agent
                )
            
            mean_reward = np.mean([sum(episode_rewards[agent]) for agent in self.env.agents])
            if episode % 50 == 0:
                print(f"Episode {episode}, Mean Reward: {mean_reward:.2f}")
            self.mean_rewards.append(mean_reward)
        return self.mean_rewards

# Training example
def main():
    env = knights_archers_zombies_v10.parallel_env(max_cycles=500)
    ppo = MultiAgentPPO(env)
    mean_rewards = ppo.train(num_episodes=1000)
    plt.plot(mean_rewards)
    plt.savefig('mappo2.png')

if __name__ == "__main__":
    main()