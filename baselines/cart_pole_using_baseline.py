__author__ = ' Zhen Wang'
import gym
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baselines import deepq
env = gym.make('CartPole-v0')
env = env.unwrapped


model = deepq.models.mlp([32, 16], layer_norm=True)
act = deepq.learn(
    env,
    # q_func=model,
    lr=0.01,
    # max_timesteps=10000,
    print_freq=1,
    checkpoint_freq=1000,
    network='mlp'
)
while True:
    obs, done = env.reset(), False
    episode_reward = 0
    while not done:
        env.render()
        obs, reward, done, _ = env.step(act(obs[None])[0])
        episode_reward += reward
    print([episode_reward, env.counts])