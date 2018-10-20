import numpy as np
from collections import defaultdict
import sys
import random
from tqdm import trange

class GridworldAgent:
    def __init__(self, env, policy, gamma = 0.9, 
                 start_epsilon = 0.9, end_epsilon = 0.1, epsilon_decay = 0.9):
        self.env = env
        self.n_action = len(self.env.action_space)
        self.policy = policy
        self.gamma = gamma
        self.v = dict.fromkeys(self.env.state_space,0)
        self.n_v = dict.fromkeys(self.env.state_space,0)
        self.q = defaultdict(lambda: np.zeros(self.n_action))
        self.n_q = defaultdict(lambda: np.zeros(self.n_action))
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay**n_episode),self.end_epsilon)
        return(epsilon)
    def get_v(self,start_state,epsilon = 0.):
        episode = self.run_episode(start_state,epsilon)
        v = np.sum([episode[i][2] * self.gamma**i for i in range(len(episode))])
        return(v)
    def get_q(self, start_state, first_action, epsilon=0.):
        episode = self.run_episode(start_state,epsilon,first_action)
        q = np.sum([episode[i][2] * self.gamma**i for i in range(len(episode))])
        return(q)
    def select_action(self,state,epsilon):
        #probs = np.ones(self.n_action) * (epsilon / self.n_action)
        #best_action = self.policy[state]
        #probs[best_action] = 1 - epsilon + (epsilon / self.n_action)
        #action = np.random.choice(np.arange(self.n_action),p=probs)
        
        best_action = self.policy[state]
        if random.random() > epsilon:
            action = best_action
        else:
             action = np.random.choice(np.arange(self.n_action))
        return(action)
    def print_policy(self):
        for i in range(self.env.sz[0]):
            print('\n----------')
            for j in range(self.env.sz[1]):
                p=self.policy[(i,j)]
                out = self.env.action_text[p]
                print(f'{out} |',end='')
    def print_v(self, decimal = 1):
        for i in range(self.env.sz[0]):
            print('\n---------------')
            for j in range(self.env.sz[1]):
                out=np.round(self.v[(i,j)],decimal)
                print(f'{out} |',end='')
    def run_episode(self, start, epsilon, first_action = None):
        result = []
        state = self.env.reset(start)
        #dictate first action to iterate q
        if first_action is not None:
            action = first_action
            next_state,reward,done,info = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: return(result)
        while True:
            action = self.select_action(state,epsilon)
            next_state,reward,done,info = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: break
        return(result)
    def update_policy_q(self):
        for state in self.env.state_space:
            self.policy[state] = np.argmax(self.q[state])
    def mc_predict_v(self,n_episode=10000,first_visit=True):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start,e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            for i in range(len(transitions)):
                if first_visit and (states[i] not in traversed):
                    traversed.append(states[i])
                    self.n_v[states[i]]+=1
                    discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                    self.v[states[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
        for state in self.env.state_space:
            if state != self.env.goal:
                self.v[state] = self.v[state] / self.n_v[state]
            else:
                self.v[state] = 0
    def mc_predict_q(self,n_episode=10000,first_visit=True):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start,e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            for i in range(len(transitions)):
                if first_visit and ((states[i],actions[i]) not in traversed):
                    traversed.append((states[i],actions[i]))
                    self.n_q[states[i]][actions[i]]+=1
                    discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                    self.q[states[i]][actions[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
                elif not first_visit:
                    self.n_q[states[i]][actions[i]]+=1
                    discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                    self.q[states[i]][actions[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
        #print(self.q,self.n_q)
        for state in self.env.state_space:
            for action in range(self.n_action):
                if state != self.env.goal:
                    self.q[state][action] = self.q[state][action] / self.n_q[state][action]
                else:
                    self.q[state][action] = 0
        
    def mc_control_q(self,n_episode=10000,first_visit=True):
        self.mc_predict_q(n_episode,first_visit)
        self.update_policy_q()
        
    def mc_control_glie(self,n_episode=10000,first_visit=True,lr=0.):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start,e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            for i in range(len(transitions)):
                if first_visit and ((states[i],actions[i]) not in traversed):
                    traversed.append((states[i],actions[i]))
                    self.n_q[states[i]][actions[i]]+=1
                    discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                    g = sum(rewards[i:]*discounts[:-(1+i)])
                    if lr > 0:
                        a = lr
                    else:
                        a = (1/self.n_q[states[i]][actions[i]])
                    self.q[states[i]][actions[i]]+= a*(g - self.q[states[i]][actions[i]])
                    self.update_policy_q()
                    
class BJAgent:
    def __init__(self, env, gamma = 1.0, 
                 start_epsilon = 1.0, end_epsilon = 0.05, epsilon_decay = 0.99999):
        
        self.env = env
        self.n_action = self.env.action_space.n
        self.policy = defaultdict(lambda: 0) #always stay as best policy
        self.v = defaultdict(lambda:0) #state value initiated as 0
        self.gamma = gamma
        
        #action values
        self.q = defaultdict(lambda: np.zeros(self.n_action)) #action value
        self.g = defaultdict(lambda: np.zeros(self.n_action)) #sum of expected rewards
        self.n_q = defaultdict(lambda: np.zeros(self.n_action)) #number of actions performed
        
        #epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
    
    #get epsilon
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return(epsilon)
    
    #select action based on epsilon greedy
    def select_action(self,state,epsilon):
        best_action = np.argmax(self.policy[state]) if state in self.q else self.env.action_space.sample()
        if random.random() > epsilon:
            action = best_action
        else:
             action = self.env.action_space.sample()
        return(action)
    
    #run episode with current policy
    def run_episode(self, epsilon):
        result = []
        state = self.env.reset()
        while True:
            action = self.select_action(state,epsilon)
            next_state,reward,done,info = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: break
        return(result)
    
    #update policy to reflect q values
    def update_policy_q(self):
        for state, value in self.q.items():
            self.policy[state] = np.argmax(value)
        
    #mc control
    def mc_control_q(self,n_episode=500000,first_visit=True,update_every=1):
        for t in trange(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            #mc prediction
            for i in range(len(transitions)):
                discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                if first_visit and ((states[i],actions[i]) not in traversed):
                    traversed.append((states[i],actions[i]))
                    self.n_q[states[i]][actions[i]]+=1
                    self.g[states[i]][actions[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
                    self.q[states[i]][actions[i]] = self.g[states[i]][actions[i]] / self.n_q[states[i]][actions[i]]
                else:
                    self.n_q[states[i]][actions[i]]+=1
                    self.g[states[i]][actions[i]]+= sum(rewards[i:]*discounts[:-(1+i)])
                    self.q[states[i]][actions[i]] = self.g[states[i]][actions[i]] / self.n_q[states[i]][actions[i]]
            #update policy every few episodes seem to be more stable
            if t % int(update_every * n_episode - 1) ==0:
                self.update_policy_q()
        #final policy update at the end
        self.update_policy_q()
    
    #mc control glie
    def mc_control_glie(self,n_episode=500000,lr=0.,update_every=1):
        for t in trange(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            
            #mc prediction
            for i in range(len(transitions)):
                discounts = np.array([self.gamma**j for j in range(len(transitions)+1)])
                traversed.append((states[i],actions[i]))
                self.n_q[states[i]][actions[i]]+=1
                g = sum(rewards[i:]*discounts[:-(1+i)])
                alpha = lr if lr > 0 else (1/self.n_q[states[i]][actions[i]])
                self.q[states[i]][actions[i]]+= alpha * (g - self.q[states[i]][actions[i]])
       
            #update policy every few episodes seem to be more stable
            if t % int(update_every * n_episode - 1)==0:
                self.update_policy_q()
        #final policy update at the end
        self.update_policy_q()
        
    #get state value from action value
    def q_to_v(self):
        for state, value in self.q.items():
            self.v[state] = np.max(value)