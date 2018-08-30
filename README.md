# Bangkok School of AI - Reinforcement Learning Workshop

## Session 1 Discrete States and Actions
* [Reinforcement learning versus other approaches](https://web.stanford.edu/class/cs234/slides/cs234_2018_l1.pdf)
* Use cases for reinforcement learning e.g. games, robotics, ads biddings, stock trading, etc.
* A word of caution:
    * [Why is reinforcement learning flawed](https://thegradient.pub/why-rl-is-flawed/)
    * [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055)
* (Windy) Gridworld example (in code and physical activities with paper and dice) to explain:
    * Problems: Markov decision processes, states, actions, and rewards
    * Solutions: policies, state values, (state-)action values, discount factor, optimality equations
* Monte Carlo Prediction
* Monte Carlo Control: all-visit, first-visit, GLIE, constant learning rate
* Challenges:
    * What are some other ways of solving reinforcement learning problems? How are they better or worse than Monte Carlo methods e.g. performance, data requirements, etc.?
    * Play around with Gridworld. Tweak these variables and see what happens:
        * Wing probability
        * Move rewards
        * Discount factor
        * Epsilon and how to decay it (or not)
    * Solve at least one of the following OpenAI gym environments with discrete states and actions:
        * FrozenLake-v0
        * Taxi-v2
        * Blackjack-v0

## Session 2 Discrete States and Actions
* OpenAI Gym toy environment to explain temporal difference learning: sarsa, q-learning, expected sarsa
* Homework: solve an environment with discrete states and actions such as:
    * FrozenLake-v0
    * Taxi-v2
    * Blackjack-v0
* Challenges: solve an environment with continuous states: discretization, tile codings, etc. such as
    * Acrobat-v1
    * MountainCar-v0
    * CartPole-v0
    * LunarLander-v2
* Points to consider:
    * What are the state space, action space, and rewards of the environment?
    * What algorithms did you use to solve the environment and why?
    * How many episodes did you solve it in? Can you improve the performance? (Tweaking discount factor, learning rate, using Monte Carlo instead of TD)
    
## Session 2.5 Neural Networks in Pytorch
* Tensor operations
* Feedforward 
* Activation functions
* Losses
* Backpropagation
* Why is deeper usually better? Spiral example

## Session 3 Continuous States and Discrete Actions
* Some approaches to continuous states: discretization, tile coding, other encoding, linear approximations
* Vanilla DQN: experience replay and target functions
* Homework: work on an Atari game and detail the process of hyperparameter tuning

## Session 4 Continuous States and Discrete Actions
* Rainbow
    * Vanilla DQN (experience replay + target network)
    * Double DQN
    * Prioritized experience replay
    * Dueling networks
    * Multi-step learning
    * Distributional RL
    * Noisy networks
* Homework: Implement Rainbow and compare it to your last project

## Session 5 Continuous States and Actions
* Policy gradient methods: a2c, a3c, ddpg, REINFORCE

## Session 6 Multi-agent Learnig
* Monte Carlo tree search

## Other Topics
* Explore vs exploit: epsilon greedy, ucb, thompson sampling
* Reward function setting
* Hackathon nights to play Blackjack, Poker, Pommerman, boardgames and self-driving cars

## Readings
* [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
* [Stanford CS234](https://web.stanford.edu/class/cs234/index.html)
* [Udacity RL Nanodegree](https://github.com/udacity/deep-reinforcement-learning)
* [David Silver Lectures](https://github.com/dalmia/David-Silver-Reinforcement-learning)
* [Denny Britz Repo](https://github.com/dennybritz/reinforcement-learning/)
* [Intro to RL in Trading](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/)
## Environments
* [OpenAI Gym](https://github.com/openai/gym)
* [Pommerman](https://github.com/suphoff/pommerman)
* [MetaCar](https://github.com/thibo73800/metacar)
* [Boardgame.io](https://github.com/google/boardgame.io)
## Agents
* [Dopamine](https://github.com/google/dopamine)