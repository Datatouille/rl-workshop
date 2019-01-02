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
    
#adapted from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs