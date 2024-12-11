import matplotlib.pyplot as plt
import os
import numpy as np


class MetricsTracker:
    def __init__(self, name=""):
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "mean_values": [],
            "mean_advantages": [],
        }
        self.dir = name

    def add_metrics(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def plot_metrics(self, save_dir="./metrics", save_individual=False):
        if self.dir != "":
            save_dir = "./metrics/" + self.dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_individual:
            for metric_name, values in self.metrics.items():
                if not values:
                    continue

                plt.figure(figsize=(10, 5))
                plt.plot(values)
                plt.title(f'{metric_name.replace("_", " ").title()}')
                plt.xlabel("Episode")
                plt.ylabel("Value")
                plt.savefig(f"{save_dir}/{metric_name}.png")
                plt.close()

        # Plot combined training metrics
        num_metrics = len(self.metrics)
        num_cols = 4
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
        )
        fig.suptitle("Training Metrics")

        for i, (metric_name, values) in enumerate(self.metrics.items()):
            row, col = divmod(i, num_cols)
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.plot(values)
            ax.set_title(metric_name.replace("_", " ").title())
            ax.set_xlabel("Episode")
            ax.set_ylabel("Value")

        # Remove any extra subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.savefig(f"{save_dir}/combined_metrics.png")
        plt.close()

    def plot_metrics_with_confidence_intervals(
        self, save_dir="./metrics", save_individual=False
    ):
        if self.dir != "":
            save_dir = "./metrics/" + self.dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_individual:
            for metric_name, values in self.metrics.items():
                if not values:
                    continue

                values = np.array(values)

                plt.figure(figsize=(10, 5))
                if values.ndim == 2:
                    mean = values.mean(axis=0)
                    stderr = values.std(axis=0) / np.sqrt(values.shape[0])
                    plt.plot(mean, label="Mean")
                    plt.fill_between(
                        range(len(mean)),
                        mean - 1.96 * stderr,
                        mean + 1.96 * stderr,
                        alpha=0.3,
                    )
                    plt.legend()
                elif values.ndim == 1:
                    plt.plot(values, label="Value")
                    plt.legend()
                plt.title(f'{metric_name.replace("_", " ").title()}')
                plt.xlabel("Episode")
                plt.ylabel("Value")
                filename = f"{save_dir}/{metric_name}_conf.png"
                plt.savefig(filename)
                plt.close()

        # Plot combined training metrics with confidence
        num_metrics = len(self.metrics)
        num_cols = 4
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
        )
        fig.suptitle("Training Metrics with Confidence Bounds")

        for i, (metric_name, values) in enumerate(self.metrics.items()):
            values = np.array(values)
            row, col = divmod(i, num_cols)
            ax = axes[row, col] if num_rows > 1 else axes[col]
            if values.ndim == 2:
                mean = values.mean(axis=0)
                stderr = values.std(axis=0) / np.sqrt(values.shape[0])
                ax.plot(mean, label="Mean")
                ax.fill_between(
                    range(len(mean)),
                    mean - 1.96 * stderr,
                    mean + 1.96 * stderr,
                    alpha=0.3,
                )
                ax.legend()
            elif values.ndim == 1:
                ax.plot(values, label="Value")
                ax.legend()
            ax.set_title(metric_name.replace("_", " ").title())
            ax.set_xlabel("Episode")
            ax.set_ylabel("Value")

        # Remove any extra subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.savefig(f"{save_dir}/combined_metrics_conf.png")
        plt.close()

    def save_metrics(self, save_dir="./metrics", name=""):
        if self.dir != "":
            save_dir = "./metrics/" + self.dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for metric_name, values in self.metrics.items():
            np.save(f"{save_dir}/{metric_name}.npy", values)

    def plot_metrics_smooth(
        self, save_dir="./metrics", save_individual=False, smoothing_window=10
    ):
        if self.dir != "":
            save_dir = "./metrics/" + self.dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def smooth(values, window_size):
            """Smooth values using a simple moving average."""
            if len(values) < window_size:
                return values
            return np.convolve(values, np.ones(window_size) / window_size, mode="valid")

        if save_individual:
            for metric_name, values in self.metrics.items():
                if not values:
                    continue

                smoothed_values = smooth(values, smoothing_window)

                plt.figure(figsize=(10, 5))
                plt.plot(values, label="Original", alpha=0.5)
                plt.plot(
                    range(len(smoothed_values)),
                    smoothed_values,
                    label="Smoothed",
                    color="orange",
                )
                plt.title(f'{metric_name.replace("_", " ").title()}')
                plt.xlabel("Episode")
                plt.ylabel("Value")
                plt.legend()
                plt.savefig(f"{save_dir}/{metric_name}_smooth.png")
                plt.close()

        # Plot combined training metrics
        num_metrics = len(self.metrics)
        num_cols = 4
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5)
        )
        fig.suptitle("Training Metrics")

        for i, (metric_name, values) in enumerate(self.metrics.items()):
            if not values:
                continue

            smoothed_values = smooth(values, smoothing_window)
            row, col = divmod(i, num_cols)
            ax = axes[row, col] if num_rows > 1 else axes[col]

            ax.plot(values, label="Original", alpha=0.5)
            ax.plot(
                range(len(smoothed_values)),
                smoothed_values,
                label="Smoothed",
                color="orange",
            )
            ax.set_title(metric_name.replace("_", " ").title())
            ax.set_xlabel("Episode")
            ax.set_ylabel("Value")
            ax.legend()

        # Remove any extra subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle
        plt.savefig(f"{save_dir}/combined_metrics_smooth.png")
        plt.close()
