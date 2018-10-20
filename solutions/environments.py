import numpy as np
from collections import defaultdict

class Gridworld:
    def __init__(self, sz = (3,3), start = (2,0), goal = (1,2), traps = [(1,1)],
                 goal_reward = 5, trap_reward = -5, move_reward = -1, wind_p = 0.):
        self.sz = sz
        self.action_text = np.array(['U','L','D','R'])
        self.action_space = np.arange(4)
        self.state_space = [(i,j) for i in range(sz[0]) for j in range(sz[1])]
        self.start =start
        self.goal = goal
        self.traps = traps
        self.move_reward = move_reward
        self.trap_reward = trap_reward
        self.goal_reward = goal_reward
        self.wind_p = wind_p
        self.reset()
    def reset(self,start=None):
        if start is None:
            self.i = self.start[0]
            self.j = self.start[1]
        else:
            self.i,self.j=start
        self.traversed = [self.start]
        self.done = False
        #physical grid
        self.physical_grid = dict.fromkeys(self.state_space,['F','x'])
        self.physical_grid[self.start] = ['F','o']
        self.physical_grid[self.goal] = ['G','x']
        for t in self.traps: self.physical_grid[t] = ['T','x']
        #reward grid
        self.reward_grid = dict.fromkeys(self.state_space,0)
        self.reward_grid[self.goal] = self.goal_reward
        for t in self.traps: self.reward_grid[t] = self.trap_reward
        return((self.i,self.j))
    def print_reward(self,visible_only=False):
        for i in range(self.sz[0]):
            print('\n----------')
            for j in range(self.sz[1]):
                if visible_only:
                    out = self.reward_grid[(i,j)] if (i,j) in self.traversed else 'NA'
                else:
                    out = self.reward_grid[(i,j)]
                print(f'{out} |',end='')
    def print_physical(self,visible_only=False):
        for i in range(self.sz[0]):
            print('\n------------------------------------')
            for j in range(self.sz[1]):
                if visible_only:
                    out = self.physical_grid[(i,j)] if (i,j) in self.traversed else ['NA','NA']
                else:
                    out = self.physical_grid[(i,j)]
                print(f'{out} |',end='')
    def update_physical(self):
        for key in self.state_space:
            self.physical_grid[key][1] = 'x'
        tile = self.physical_grid[(self.i,self.j)][0] 
        self.physical_grid[(self.i,self.j)] = [tile,'o']
    def wind(self):
        offset = np.random.choice([-1,1])
        if np.random.uniform() < self.wind_p:
            if np.random.uniform() < 0.5:
                pos = self.i + offset
                self.i = np.clip(pos,0,self.sz[0]-1)
            else:
                pos = self.j + offset
                self.j = np.clip(pos,0,self.sz[1]-1)
            
    def step(self,action):
        reward = self.move_reward
        i,j = self.i,self.j
        action = self.action_text[action]
        if action == 'U':
            i -= 1
        elif action == 'L':
            j -= 1
        elif action == 'D':
            i += 1
        elif action == 'R':
            j += 1
        #check legality
        if (i,j) in self.state_space:
            #update position
            self.i,self.j = i,j
            #wind blows
            self.wind()
            #save traversed
            self.traversed.append((self.i,self.j))
            #update physical
            self.update_physical()
            #update reward
            reward += self.reward_grid[(self.i,self.j)]
        else:
            pass
        if (self.i,self.j) == self.goal: self.done = True
        #return s',r, done or not
        return((self.i,self.j),reward,self.done,'info')