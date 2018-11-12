__author__ = ' Zhen Wang'
import numpy as np
import pandas as pd
import time

num_states = 6
actions = ['left', 'right']
epsilon = 0.9
alpha = 0.1
gamma = 0.9
max_episodes = 10
refresh_interval = 0.3


class Q_table(object):

    def __init__(self):
        self.table = self._build_q_table()

    def _build_q_table(self):
        table = pd.DataFrame(np.zeros((num_states, len(actions))), columns=actions,)
        return table


class Env(object):
    def __init__(self, episode):
        self.num_states = 6
        self.actions = ['left', 'right']
        self.update_env(0, episode, 0)


    def get_env_feedback(self, S, A):
        if A =='right':
            if S == num_states-2:
                S1 = 'terminal'
                reward = 1
            else:
                S1 = S + 1
                reward = 0
        else:
            reward = 0
            if S ==0:
                S1 = S
            else:
                S1 = S - 1
        return S1 , reward

    def update_env(self, S, episode, step_counter):
        self.env_list = ['-'] *(num_states-1) + [ 'T']
        if S == 'terminal':
            print('\r{}'.format("episode {}".format(episode+1)), "steps {}".format(step_counter))
            time.sleep(1)
        else:
            self.env_list[S] = 'o'
            print(''.join(self.env_list))
            time.sleep(1)


class Agent(object):
    def __init__(self):
        self.state = 0

    def choose_action(self, q_table):
        state_actions = q_table.iloc[self.state, :]
        if (np.random.uniform(0,1) > epsilon) or ((state_actions == 0).all()):
            action_name = np.random.choice(actions)
        else:
            action_name = state_actions.idxmax()
        return action_name


def q_learning():
    q_table = Q_table()
    for episode in range(max_episodes):
        agent = Agent()

        env = Env(episode=episode)
        step_counter = 0
        is_terminated = False
        while not is_terminated:
            A = agent.choose_action(q_table.table)
            S_next, reward = env.get_env_feedback(agent.state, A)
            q_predict =q_table.table.loc[agent.state, A]
            if S_next != 'terminal':
                q_target = reward + gamma * q_table.table.iloc[S_next, :].max()
            else:
                q_target = reward
                is_terminated = True
            q_table.table.loc[agent.state, A] += alpha * (q_target - q_predict)
            agent.state = S_next
            step_counter +=1
            env.update_env(agent.state, episode, step_counter)

    print(q_table.table)


def sarsa():
    q_table = Q_table()
    for episode in range(max_episodes):
        agent = Agent()
        env = Env(episode=episode)
        step_counter = 0
        is_terminated = False
        A = agent.choose_action(q_table.table)
        while not is_terminated:
            S = agent.state
            S_next, reward = env.get_env_feedback(agent.state, A)
            q_predict =q_table.table.loc[agent.state, A]
            agent.state = S_next
            if S_next != 'terminal':
                A_next =  agent.choose_action(q_table.table)
                q_target = reward + gamma * q_table.table.loc[S_next, A_next]
            else:
                q_target = reward
                is_terminated = True
            q_table.table.loc[S, A] += alpha * (q_target - q_predict)
            A = A_next
            step_counter +=1
            env.update_env(agent.state, episode, step_counter)

    print(q_table.table)

if __name__ == '__main__':
    sarsa()