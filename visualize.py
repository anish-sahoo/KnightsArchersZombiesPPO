import numpy as np
import os

import matplotlib.pyplot as plt

# Specify the folder path
folder_path = r'metrics\run6'

# Load the data
entropies = np.load(os.path.join(folder_path, 'entropies.npy'))
episode_rewards = np.load(os.path.join(folder_path, 'episode_rewards.npy'))
policy_losses = np.load(os.path.join(folder_path, 'policy_losses.npy'))
value_losses = np.load(os.path.join(folder_path, 'value_losses.npy'))

# Create a 2x2 plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot entropies
axs[0, 0].plot(entropies)
axs[0, 0].set_title('Entropies')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Entropy')

# Plot episode rewards
axs[0, 1].plot(episode_rewards)
axs[0, 1].set_title('Episode Rewards')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Reward')

# Plot policy losses
axs[1, 0].plot(policy_losses)
axs[1, 0].set_title('Policy Losses')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Loss')

# Plot value losses
axs[1, 1].plot(value_losses)
axs[1, 1].set_title('Value Losses')
axs[1, 1].set_xlabel('Episode')
axs[1, 1].set_ylabel('Loss')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()