# Updated PopArt.py
import numpy as np

class PopArt:
    def __init__(self, beta=0.0003, epsilon=1e-5):
        self.mean = 0.0
        self.sq_mean = 0.0
        self.beta = beta
        self.epsilon = epsilon

    def normalize(self, x):        
        x = np.array(x)
        self.mean = self.beta * np.mean(x) + (1 - self.beta) * self.mean
        self.sq_mean = self.beta * np.mean(x ** 2) + (1 - self.beta) * self.sq_mean
        std = np.sqrt(self.sq_mean - self.mean ** 2 + self.epsilon)
        return (x - self.mean) / std

    def update(self, x):
        """Update normalization statistics with new data."""
        x = np.array(x)
        self.mean = self.beta * np.mean(x) + (1 - self.beta) * self.mean
        self.sq_mean = self.beta * np.mean(x ** 2) + (1 - self.beta) * self.sq_mean

    def denormalize(self, y):
        """Convert normalized data back to original scale."""
        std = np.sqrt(self.sq_mean - self.mean ** 2 + self.epsilon)
        return y * std + self.mean