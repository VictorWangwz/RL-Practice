__author__ = ' Zhen Wang'
import numpy as np
import tensorflow as tf
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
np.random.seed(1)
tf.set_random_seed(1)


class DQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 alpha=0.01,
                 gamma=0.9,
                 epsilon=1,
                 replace_target_iter=200,
                 memory_size=2000,
                 batch_size=32,
                 epsilon_increment=None,
                 output_graph=False
                 ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.alpah = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.step_counter = 0

        self.memory = np.zeros((self.memory_size, self.n_features * 2 +2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.alpah).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('target q updated')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval], feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s : batch_memory[:, :self.n_features]
            }
        )
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                         self.s:batch_memory[:, :self.n_features],
                                         self.q_target: q_target
                                     })
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.step_counter += 1

    def plot_cost(self):

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()





