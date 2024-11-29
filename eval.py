import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=20,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=30,
    max_cycles=500,
)

# Load the trained weights
actor.load_state_dict(torch.load("actor_weights.pth"))
critic.load_state_dict(torch.load("critic_weights.pth"))

# Run 10 games and collect rewards
num_games = 10
game_rewards = []

for _ in range(num_games):
    env.reset()
    total_reward = 0
    while env.agents:
        actions = {}
        for agent in env.agents:
            if agent not in observations:
                obs = torch.zeros((512, 512, 3), dtype=torch.float32).unsqueeze(0).to(device)
            else:
                obs = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = actor(obs)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            actions[agent] = action.item()

        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent in env.agents:
            total_reward += rewards[agent]
        if all(terminations.values()) or all(truncations.values()):
            break
    game_rewards.append(total_reward)

# Plot the sum of rewards
plt.figure()
plt.plot(game_rewards, label="Sum of Rewards")
plt.xlabel("Game")
plt.ylabel("Total Reward")
plt.legend()
plt.savefig("sum_rewards.png")