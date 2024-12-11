import numpy as np
import os
import matplotlib.pyplot as plt


def plot_metrics(folder_path, name, window_size=100):
    entropies = np.load(os.path.join(folder_path, "entropies.npy"))
    episode_rewards = np.load(os.path.join(folder_path, "episode_rewards.npy"))
    episode_rewards = np.clip(episode_rewards, -100, 30)

    running_avg = np.convolve(
        episode_rewards, np.ones(window_size) / window_size, mode="valid"
    )

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle(name)

    axs[0].plot(entropies)
    axs[0].set_title("Entropies")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Entropy")

    axs[1].plot(episode_rewards, label="Episode Rewards")
    axs[1].plot(
        range(window_size - 1, len(episode_rewards)),
        running_avg,
        label=f"Running Average ({window_size})",
        color="orange",
    )
    axs[1].set_title("Episode Rewards")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Reward")
    axs[1].legend()

    plt.tight_layout()

    plt.savefig(f"metrics_{name}.png")

    return running_avg


def plot_combined_smoothed_rewards(arrays, names, window_size=100):
    fig, ax = plt.subplots(figsize=(4, 4))
    for data, name in zip(arrays, names):
        ax.plot(data, label=name)
    ax.set_title("Episodic Rewards Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend()
    plt.tight_layout()
    plt.savefig("combined_rewards.png")


window_size = 500
run6_avg = plot_metrics("metrics/run6", "Initial Training Run", window_size)
run12_avg = plot_metrics("metrics/run12", "Training Run with Framestack", window_size)

plot_combined_smoothed_rewards(
    [run6_avg, run12_avg], ["Initial Run", "Framestack"], window_size
)
