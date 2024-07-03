# RLandGym

My GYM for reinforcement learning.

I use the environment [Gymnasium](https://gymnasium.farama.org/) from OpenAI to train the reinforcement learning model and also to *reinforce* my coding skills.

## Files

1. Agent: Contains the implementation of all agents.
    1. utils.py: Contains *Logging* and *Replaybuffer*.
    2. a2c.py: Implementation of *A2C* in *CartPole-v1*.
    3. ppo.py: Implementation of *PPO* in *CartPole-v1*. There is [implementation of OpenAI](https://github.com/openai/baselines/tree/master/baselines/ppo2) and [implementation of Morvan](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/discrete_DPPO.py#L134).
2. .gitignore: The files and dirs should be ignored by *Git*.
