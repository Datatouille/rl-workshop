from collections import defaultdict, namedtuple, deque
import random
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state','done'))
class VanillaMemory:
    def __init__(self, capacity, seed = 1412):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity) 
        self.seed = random.seed(seed)
    def add(self, *args):
        t = Transition(*args)
        self.memory.append(t)
    def sample(self, batch_size):
        ts = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([t.state for t in ts])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in ts])).float().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in ts])).float().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in ts])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in ts]).astype(np.uint8)).float().to(device)
        return(states,actions,rewards,next_states,dones)
    def __len__(self):
        return(len(self.memory))
    

class PrioritizedMemory:
    """
    https://arxiv.org/pdf/1511.05952.pdf
    """
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.alpha = alpha
        self.priority = deque(maxlen=capacity)
        self.probs = np.zeros(capacity)
    def add(self, *args):
        max_prior = max(self.priority) if self.memory else 1.
        t = Transition(*args)
        self.memory.append(t)
        #give latest transition max priority for optimistic start
        self.priority.append(max_prior)
    def prior_to_prob(self):
        probs = np.array([i**self.alpha for i in self.priority]) #uniform sampling when alpha is 0
        self.probs[range(len(self.priority))] = probs
        self.probs /= np.sum(self.probs)
    def sample(self, batch_size, beta = 0.4):
        #calculate prob every time we will sample
        self.prior_to_prob()
        idx = np.random.choice(range(self.capacity), batch_size, replace=False, p=self.probs)
        ts = [self.memory[i] for i in idx]
        
        #stitch tuple
        states = torch.from_numpy(np.vstack([t.state for t in ts])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in ts])).float().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in ts])).float().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in ts])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in ts]).astype(np.uint8)).float().to(device)
        
        #importance sampling weights
        sampling_weights = (len(self.memory)*self.probs[idx])**(-beta) #higher beta, higher compensation for prioritized sampling
        sampling_weights = sampling_weights / np.max(sampling_weights) #normalize by max weight to always scale down
        sampling_weights = torch.from_numpy(sampling_weights).float().to(device)
        
        return(states,actions,rewards,next_states,dones,idx,sampling_weights)
    def update_priority(self,idx,losses):
        for i, l in zip(idx, losses):
            self.priority[i] = l.data
        
    def __len__(self):
        return(len(self.memory))
    