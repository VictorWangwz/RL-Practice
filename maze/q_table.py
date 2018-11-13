__author__ = ' Zhen Wang'

import numpy as np
import pandas as pd


class QTable(object):
    def __init__(self, actions, alpha=0.01, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            state_action = self.table.loc[state,:]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append(pd.Series([0] * len(self.actions),
                                                     index=self.table.columns,
                                                     name=state))

    def learn(self,s,a,r, s_next):
        self.check_state_exist(s_next)
        q_predict = self.table.loc[s,a]
        if s_next !='terminal':
            q_target = r + self.gamma * self.table.loc[s_next, :].max()
        else:
            q_target = r
        self.table.loc[s,a] += self.alpha * (q_target - q_predict)


