import matplotlib.pyplot as plt
import os
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'mean_values': [],
            'mean_advantages': [],
            'buffer_capacity_used': []
        }
    
    def add_metrics(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
    def plot_metrics(self, save_dir='./metrics', save_individual=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if save_individual:
            for metric_name, values in self.metrics.items():
                if not values:
                    continue
                    
                plt.figure(figsize=(10, 5))
                plt.plot(values)
                plt.title(f'{metric_name.replace("_", " ").title()}')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.savefig(f'{save_dir}/{metric_name}.png')
                plt.close()
            
        # Plot combined training metrics
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Training Metrics')
        
        metrics_to_plot = ['episode_rewards', 'policy_losses', 'value_losses', 'entropies']
        titles = ['Episode Rewards', 'Policy Loss', 'Value Loss', 'Entropy']
        y_labels = ['Rewards', 'Loss', 'Loss', 'Entropy']
        
        for i, metric_name in enumerate(metrics_to_plot):
            axes[i].plot(self.metrics[metric_name])
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(y_labels[i])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/combined_metrics.png')
        plt.close()
        
    def plot_metrics_with_confidence_intervals(self, save_dir='./metrics', save_individual=False, confidence=0.95):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        def compute_confidence_interval(data, confidence=0.95):
            n = len(data)
            mean = np.mean(data)
            se = np.std(data) / np.sqrt(n)
            h = se * 1.96  # for 95% confidence
            return mean, mean - h, mean + h
        
        if save_individual:
            for metric_name, values in self.metrics.items():
                if not values:
                    continue
                
                means, lower_bounds, upper_bounds = [], [], []
                for i in range(len(values)):
                    mean, lower, upper = compute_confidence_interval(values[:i+1], confidence)
                    means.append(mean)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                
                plt.figure(figsize=(10, 5))
                plt.plot(values, label='Values')
                plt.plot(means, label='Mean')
                plt.fill_between(range(len(values)), lower_bounds, upper_bounds, color='b', alpha=0.2, label='Confidence Interval')
                plt.title(f'{metric_name.replace("_", " ").title()}')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.legend()
                plt.savefig(f'{save_dir}/{metric_name}_ci.png')
                plt.close()
        
        # Plot combined training metrics with confidence intervals
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Training Metrics with Confidence Intervals')
        
        metrics_to_plot = ['episode_rewards', 'policy_losses', 'value_losses', 'entropies']
        titles = ['Episode Rewards', 'Policy Loss', 'Value Loss', 'Entropy']
        y_labels = ['Rewards', 'Loss', 'Loss', 'Entropy']
        
        for i, metric_name in enumerate(metrics_to_plot):
            values = self.metrics[metric_name]
            if not values:
                continue
            
            means, lower_bounds, upper_bounds = [], [], []
            for j in range(len(values)):
                mean, lower, upper = compute_confidence_interval(values[:j+1], confidence)
                means.append(mean)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            axes[i].plot(values, label='Values')
            axes[i].plot(means, label='Mean')
            axes[i].fill_between(range(len(values)), lower_bounds, upper_bounds, color='b', alpha=0.2, label='Confidence Interval')
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Episode')  # Ensure this line is executed
            axes[i].set_ylabel(y_labels[i])  # Ensure this line is executed
            axes[i].legend()
                
        plt.tight_layout()
        plt.savefig(f'{save_dir}/combined_metrics_ci.png')
        plt.close()
