from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def total(self):
        return len(self.memory)
    
    def pointer_sample(self, pointer):
        ran = random.randint(self.total()-pointer,self.total() - 1)
        return self.memory[ran]
    
    # def add(child_queue)
    
