import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    
    def __init__(self, action_dim, device):
        super().__init__()
        self.device = device
        # Input: 84x84 grayscale
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)  # Output: 20x20x32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # Output: 9x9x64
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # Output: 7x7x64
        
        # Calculate flattened size: 7 * 7 * 64 = 3136
        self.fc = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        
        self.to(self.device)
        
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
        if x.device != self.device:
            x = x.to(self.device)
        
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