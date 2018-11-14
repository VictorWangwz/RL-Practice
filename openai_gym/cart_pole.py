__author__ = ' Zhen Wang'
import gym
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from maze.DQN import DQN
env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

nn = DQN(env.action_space.n, env.observation_space.shape[0],
         alpha=0.01,
         gamma=0.9,
         epsilon=0.9,
         replace_target_iter=100,
         memory_size=2000,
         epsilon_increment=0.01)



total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()
        action = nn.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        nn.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            nn.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(nn.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

nn.plot_cost()