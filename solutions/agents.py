#misc
import numpy as np
from collections import defaultdict
import sys
import random
from tqdm import trange

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#network and utility
from solutions.networks import *
from solutions.utils import *

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
        best_action = self.policy[state] if state in self.q else self.env.action_space.sample()
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
            
class TaxiAgent:
    def __init__(self, env, gamma = 0.8, alpha = 1e-1,
                 start_epsilon = 1, end_epsilon = 1e-2, epsilon_decay = 0.999):
        
        self.env = env
        self.n_action = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        
        #action values
        self.q = defaultdict(lambda: np.zeros(self.n_action)) #action value
        
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
        #implicit policy; if we have action values for that state, choose the largest one, else random
        best_action = np.argmax(self.q[state]) if state in self.q else self.env.action_space.sample()
        if random.random() > epsilon:
            action = best_action
        else:
             action = self.env.action_space.sample()
        return(action)
    
    def sarsa_update(self, state, action, reward, next_state, n_episode):
        #get next action
        next_action = self.select_action(next_state, self.get_epsilon(n_episode)) 
        #get new q
        new_q = reward + (self.gamma * self.q[next_state][next_action])
        #calculate update equation
        self.q[state][action] = self.q[state][action] + (self.alpha * (new_q - self.q[state][action]))
        
    def sarsa_max_update(self, state, action, reward, next_state):
        #get new q
        new_q = reward + (self.gamma * np.max(self.q[next_state]))
        #calculate update equation
        self.q[state][action] = self.q[state][action] + (self.alpha * (new_q - self.q[state][action]))
        
    def sarsa_expected_update(self, state, action, reward, next_state, n_episode):
        #get next action
        next_action = self.select_action(next_state, self.get_epsilon(n_episode)) 
        #get current epsilon
        eps = self.get_epsilon(n_episode)
        #get q value of random portion
        random_q = eps * np.sum((1/self.n_action) * self.q[next_state])
        #get q value of best action
        best_q = (1-eps) * self.q[next_state][next_action]
        #get new q
        new_q = reward + self.gamma * (random_q+best_q)
        #calculate update equation
        self.q[state][action] = self.q[state][action] + (self.alpha * (new_q - self.q[state][action]))

class PGAgent:
    def __init__(self, env, state_size, action_size, hidden_size, lr = 1e-2):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.policy = PolicyNetwork(state_size,action_size,hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def play_episode(self, t_max=2000, nb_episode = 1):
        
        #collection of trajectories
        trajectories = []

        for e in range(nb_episode):
            #each trajectory
            states= []
            actions = []
            log_probs = []
            rewards = []
            dones = []

            #initialize state
            state = self.env.reset()
            for t in range(t_max):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                #get action and log prob
                dist = self.policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action) #do not detach from graph as we need this for computing loss
                action = action.item()

                #environment step
                next_state, reward, done, info = env.step(action)

                #tuple
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)

                #advance to next state
                state = next_state

                #break if done
                if done: break

            #create and append trajectories
            trajectory = (states,actions,log_probs,rewards,dones)
            trajectories.append(trajectory)

        return(trajectories)


    def discounted_rewards(self, rewards, dones, gamma=0.99, normalized=True):
        '''
        Get discounted rewards
        '''
        g = 0
        returns = []
        #iterate from the last reward first; e.g. rewards = [0,1,2]
        #we iterate from 2, 1, 0
        for i in reversed(range(len(rewards))):
            #calculate discounted reward at timestep i
            #discounted reward = current reward + gamma * discounted reward from next timestep i+1
            #check if it's done then don't add up future discounted reward 
            g = rewards[i] + gamma * g * (1-dones[i])
            #insert this in the front of the array (since we are doing things in reverse)
            returns.insert(0, g)

        #normalize rewards
        if normalized:
            returns_avg = np.mean(returns)
            returns_std = np.std(returns)
            returns_std = max(1e-8, returns_std)
            returns_normalized = (returns - returns_avg) / (returns_std)
            returns = returns_normalized
        return(returns)
    
    def act(self, state):
        '''
        Get stochastic action based on state
        '''
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        dist = self.policy(state_tensor)
        action = dist.sample().item()
        return(action)

    def surrogate(self, rewards, dones, log_probs, gamma = 0.99, normalized=True):
        '''
        Get surrogate function to optimize by gradient ascent
        '''
        #get discounted rewards
        gs = self.discounted_rewards(rewards, dones, gamma, normalized)

        #loss is (log) probability of doing an action times its discounted (normalized) rewards
        loss = log_probs * gs
        
        #minus sign for telling pytorch we are doing gradient **ascent**
        return(-loss.sum())
    
    def train(self, t_max = 2000, nb_episode=1):
        '''
        Train for a number of episodes
        '''
        scores = 0
        losses = 0
        #collect trajectories
        for trajectory in self.play_episode(t_max=t_max, nb_episode=nb_episode):
            states, actions, log_probs, rewards, dones = trajectory

            #calculate loss
            loss = self.surrogate(rewards,dones, log_probs, gamma=0.99, normalized=True)
            #run optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            scores += sum(rewards)
            losses += loss
        
        #average across episodes learned
        scores /= nb_episode
        losses /= nb_episode
        
        return(losses,scores)
                
    
class DQNAgent:
    def __init__(self, state_size = 2, action_size = 3, replay_memory = None, seed = 1412,
        lr = 1e-3 / 4, bs = 64, nb_hidden = 256, clip = 1.,
        gamma=0.99, tau= 1e-3, update_interval = 5, update_times = 1, tpe = 200):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.npseed = np.random.seed(seed)
        self.lr = lr
        self.bs = bs
        self.gamma = gamma
        self.update_interval = update_interval
        self.update_times = update_times
        self.tau = tau
        self.losses = []
        self.tpe = tpe
        self.clip = clip

        #vanilla
        self.network_local = QNetwork(state_size, action_size, nb_hidden).to(device)
        self.network_target = QNetwork(state_size, action_size, nb_hidden).to(device)
        
        #dueling
#         self.network_local = DuelingNetwork(state_size, action_size, nb_hidden).to(device)
#         self.network_target = DuelingNetwork(state_size, action_size, nb_hidden).to(device)
        
        #optimizer
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=self.lr)

        # replay memory
        self.memory = replay_memory
        # count time steps
        self.t_step = 0
        
    def get_eps(self, i, eps_start = 1., eps_end = 0.001, eps_decay = 0.9):
        eps = max(eps_start * (eps_decay ** i), eps_end)
        return(eps)
    
    def step(self, state, action, reward, next_state, done):
        #add transition to replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        #update target network
        self.soft_update(self.network_local, self.network_target)
#         self.hard_update(self.network_local, self.network_target)
        
        # learn every self.t_step
        self.t_step += 1
        if self.t_step % self.update_interval == 0:
            if len(self.memory) > self.bs:
                #vanilla
                for _ in range(self.update_times):
                    transitions = self.memory.sample(self.bs)
                    self.learn(transitions)

    def act(self, state):
        eps = self.get_eps(int(self.t_step / self.tpe))
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(state)
        self.network_local.train()

        #epsilon greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def vanilla_loss(self,q_targets,q_expected):
        loss = F.mse_loss(q_expected,q_targets)
        return(loss)
        
    def learn(self, transitions, small_e = 1e-5):
        #vanilla
        states, actions, rewards, next_states, dones = transitions

        #vanilla
        #         q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        #double
        max_actions_next = self.network_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.network_target(next_states).detach().gather(1, max_actions_next.long())

        #compute loss
        q_targets = rewards + (self.gamma * q_targets_next) * (1 - dones)
        q_expected = self.network_local(states).gather(1, actions.long())
        #vanilla
        loss = self.vanilla_loss(q_expected, q_targets)
        #append for reporting
        self.losses.append(loss)
        
        #backprop
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip: torch.nn.utils.clip_grad_norm(self.network_local.parameters(), self.clip)
        self.optimizer.step()
      
    def hard_update(self, local_model, target_model):
        if self.t_step % 1/self.tau==0:
            target_model.load_state_dict(local_model.state_dict())
            
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class DDPGAgent():        
    def __init__(self, 
        state_size, action_size, nb_hidden, replay_memory, action_bounds = [-2,2], random_seed=0, 
        bs = 128, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-4, wd_actor=0, wd_critic=0,
        clip_actor = None, clip_critic=None, update_interval = 20, update_times = 10): 

        self.state_size = state_size
        self.action_size = action_size
        self.action_lower = action_bounds[0]
        self.action_upper = action_bounds[1]
        self.seed = random.seed(random_seed)
        self.bs = bs
        self.update_interval = update_interval
        self.update_times = update_times
        self.timestep = 0

        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.wd_critic = wd_critic
        self.wd_actor = wd_actor
        self.clip_critic=clip_critic
        self.clip_actor = clip_actor
        self.actor_losses = []
        self.critic_losses = []

        #actor
        self.actor_local = ActorNetwork(state_size, action_size, nb_hidden, random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, nb_hidden, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor,weight_decay=self.wd_actor)

        #critic
        self.critic_local = CriticNetwork(state_size, action_size, nb_hidden, random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, nb_hidden, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic,weight_decay=self.wd_critic)

        #noise
        self.noise = OUNoise(action_size, random_seed)

        #replay memory
        self.memory = replay_memory
    
    def step(self, state, action, reward, next_state, done):
        #increment timestep
        self.timestep+=1
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory  
        if self.timestep % self.update_interval == 0:
            for i in range(self.update_times):
                if len(self.memory) > self.bs:
                    transitions = self.memory.sample(self.bs)
                    self.learn(transitions)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()        
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, self.action_lower, self.action_upper)

    def reset_noise(self):
        self.noise.reset()

    def learn(self, transitions):
        states, actions, rewards, next_states, dones = transitions

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_critic: torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), self.clip_critic)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_actor: torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), self.clip_actor)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)   
           
        self.actor_losses.append(actor_loss.cpu().data.numpy())
        self.critic_losses.append(critic_loss.cpu().data.numpy())        

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        if self.t_step % 1/self.tau==0:
            target_model.load_state_dict(local_model.state_dict())