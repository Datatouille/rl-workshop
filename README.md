# Bangkok School of AI - Reinforcement Learning Workshop

## How to Use Notebooks

Each notebook contains the content and code-along of each session. We recommend that you run the notebooks from [Google Colaboratory](https://colab.research.google.com/) for minimal setup requirements. Edit the `Fill in The Code` section for coding assigments and check with our way of solving them in `solutions`.

## Session 1 Escaping GridWorld with Simple RL Agents
*Markov Decision Processes / Discrete States and Actions*
* What is Reinforcement Learning: Pavlov's kitties
* How Useful is Reinforcement Learning: games, robotics, ads biddings, stock trading, etc.
* Why is Reinforcement Learning Different: level of workflow automation in classes of machine learning algorithm
    * Use cases for reinforcement learning 
* Reinforcement Learning Framework and Markov Decision Processes
* GridWorld example to explain:
    * Problems: Markov decision processes, states, actions, and rewards
    * Solutions: policies, state values, (state-)action values, discount factor, optimality equations
* Words of Caution: a few reasons [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
* Challenges:
    * Read up on Bellman's equations and find out where they hid in our workshop today.
    * What are you ideas about how we can find the policy policy?
    * Play around with Gridworld. Tweak these variables and see what happens to state and action values:
        * Expand the grid and/or add some more traps
        * Wind probability
        * Move rewards
        * Discount factor
        * Epsilon and how to decay it (or not)

## Session 2 Win Big at Monte Carlo - Sponsored by Humanize, the company that helps your business grow with AI
*Discrete States and Actions*
* Blackjack-v0 environment, human play and computer play
* Optimal Strategy for Blackjack
* What is Monte Carlo Method
* Monte Carlo Prediction
* Monte Carlo Control: All-visit, First-visit, and GLIE
* Challanges:
    * What are some other ways of solving reinforcement learning problems? How are they better or worse than Monte Carlo methods e.g. performance, data requirements, etc.?
    * Solve at least one of the following OpenAI gym environments with discrete states and actions:
        * FrozenLake-v0
        * Taxi-v2
        * Blackjack-v0
        * Any other environments with discrete states and actions at [OpenAI Gym](https://github.com/openai/gym/wiki/Table-of-environments)
    * Check `session2b.ipynb` if you are interested in using Monte Carlo method to solve Grid World. This will give you more insight into difference between all-visit and first-visit Monte Carlo.

## Session 3 GET a Taxi with Temporal Difference Learning - Sponsored by GET, the new ride-hailing service in Thailand
*Discrete States and Actions*
* Taxi-v2 environment
* Comparison between Monte Carlo and TD
* SARSA
* Q-learning
* Expected SARSA
* Handling Continuous States
* Challenges: Solve an environment with continuous states using discretization
    * Acrobat-v1
    * MountainCar-v0
    * CartPole-v0
    * LunarLander-v2
* Points to consider:
    * What are other ways of handling continuous states? (See tile coding)
    * What are the state space, action space, and rewards of the environment?
    * What algorithms did you use to solve the environment and why?
    * How many episodes did you solve it in? Can you improve the performance? (Tweaking discount factor, learning rate, Monte Carlo vs TD)
    
## Session 3b Neural Networks in Pytorch - Sponsored by GET, the new ride-hailing service in Thailand
*Optional*
1. **Building Blocks** 

Familiarize ourselves with basic building blocks of a neural network in PyTorch such as tensors and layers

2. **Your First Neural Network**

Build your first neural network with the main components of architecture, loss and optimizer

3. **Spiral Example** 

Use your first neural network in a task challenging for linear models to understand why we even need deep learning

## Session 4 Policy-gradient Pong - Sponsored by GET, the new ride-hailing service in Thailand
*Continuous States and Discrete Actions*
* Replacing Q dictionaries with neural networks
* Pong-v0 environment
* Monte Carlo Policy Gradient aka Vanilla Policy Gradient aka REINFORCE aka Stochastic Policy Gradient
* Proximal Policy Optimization (PPO)
* Challenges: 
   * Train your own PPO agent to play Pong-v0
   * Try better policy network such as incorporating `nn.Conv2d`

## Session 5 Deep Deep Q-learning to Drive MountainCar - Sponsored by GET, the new ride-hailing service in Thailand
*Continuous States and Discrete Actions*
* MountainCar-v0 environment
* Deep Q-learning (DQN)
* Train Your Own DQN Agent:
    * Hyperparameter tuning
    * Reward engineering
* Inside DQN Agent:
    * Replay Memory
    * Q Networks
    * Agent action selection
    * Agent update: DQN and DDQN
* Challenges:
    * Finetune the model and try to beat [OpenAI Leaderboard](https://github.com/openai/gym/wiki/Leaderboard) at 341 episodes. Use what you learn from this session such as creative reward engineering and other hyperparameter tunings.
    * Try to figure out how you can solve `MountainCarContinuous-v0`. It is almost exactly the same as `MountainCar-v0` but with continuous action space of size 1. See [NAF Q-learning](https://arxiv.org/abs/1603.00748) and [DDPG](https://arxiv.org/abs/1509.02971) papers for some hints.
    * Read up on [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) and how to push DQN to its limits.

## Session 5.5 Rainbow
*Continuous States and Discrete Actions*
* Rainbow
    * Vanilla DQN (experience replay + target network)
    * Double DQN
    * Prioritized experience replay
    * Dueling networks
    * Multi-step learning
    * Distributional RL
    * Noisy networks
* Take-home Challenges: Implement Rainbow and compare it to your last project

## Session 6 Actor-Critic Methods
* Policy-based vs value-based deep reinforcement learning
* Vanilla Actor-Critic
* DDPG

## Session 6.5 Advanced Actor-Critic Methods
* A2C
* A3C
* SAC


## Other Topics
* Explore vs exploit: epsilon greedy, ucb, thompson sampling
* Reward function setting
* Monte Carlo Tree Search
* Hackathon nights to play Blackjack, Poker, Pommerman, boardgames and self-driving cars

## Readings
* [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
* [Stanford CS234](https://web.stanford.edu/class/cs234/index.html)
* [Udacity RL Nanodegree](https://github.com/udacity/deep-reinforcement-learning)
* [David Silver Lectures](https://github.com/dalmia/David-Silver-Reinforcement-learning)
* [UC Berley Lectures](http://rail.eecs.berkeley.edu/deeprlcourse/)
* [Siraj's Move 37](https://www.theschool.ai/courses/move-37-course/)
* [Denny Britz Repo](https://github.com/dennybritz/reinforcement-learning/)
* [Intro to RL in Trading](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/)

## Environments
* [Spinning Up](https://github.com/openai/spinningup) - an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL)
* [OpenAI Gym](https://github.com/openai/gym) - a toolkit for developing and comparing reinforcement learning algorithms
* [Unity ML-Agent Toolkit](https://github.com/Unity-Technologies/ml-agents) - an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents
* [Holodeck](https://github.com/byu-pccl/holodeck) - a high-fidelity simulator for reinforcement learning built on top of Unreal Engine 4
* [AirSim](https://github.com/Microsoft/AirSim) - a simulator for drones, cars and more, built on Unreal Engine
* [Carla](https://github.com/carla-simulator/carla) - an open-source simulator for autonomous driving research
* [Pommerman](https://github.com/suphoff/pommerman) - a clone of Bomberman built for AI research
* [MetaCar](https://github.com/thibo73800/metacar) - a reinforcement learning environment for self-driving cars in the browser
* [Boardgame.io](https://github.com/google/boardgame.io) - a boardgame environment

## Agents
* [Unity ML-Agent Toolkit](https://github.com/Unity-Technologies/ml-agents) - an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents
* [SLM Labs](https://github.com/kengz/SLM-Lab) - a modular deep reinforcement learning framework in PyTorch
* [Dopamine](https://github.com/google/dopamine) - a research framework for fast prototyping of reinforcement learning algorithms
* [TRF](https://github.com/deepmind/trfl/) - a library built on top of TensorFlow that exposes several useful building blocks for implementing Reinforcement Learning agent
* [Horizon](https://github.com/facebookresearch/Horizon) - an open source end-to-end platform for applied reinforcement learning (RL) developed and used at Facebook. 
