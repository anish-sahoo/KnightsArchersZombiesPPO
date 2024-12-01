class ReplayBuffer:
    """
    Replay buffer for storing past experiences allowing the agent to learn from them.
    
    Args:
    max_size (int): Maximum number of experiences that can be stored in the buffer
    """
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