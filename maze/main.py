__author__ = ' Zhen Wang'
from maze_env import Maze
from q_table import QTable
from maze_env_dqn import Maze as Maze_dqn
from DQN import DQN


def q_learning_update():
    for episode in range(100):
        state = env.reset()
        while True:
            env.render()
            action = table.choose_action(str(state))
            state_next, reward, done = env.step((action))
            table.learn(str(state), action, reward, str(state_next))
            state = state_next
            if done:
                break
        print(table.table)
    print('over')
    env.destroy()

def nn_update():
    step = 0
    for episode in range(300):
        state = env1.reset()
        while True:
            env1.render()
            action = nn.choose_action(state)
            state_, reward, done = env1.step(action)
            nn.store_transition(state, action, reward, state_)
            if step > 200 and step%5 ==0:
                nn.learn()
            state = state_
            if done:
                break
            step += 1
    print('over')
    env.destroy

def q_learning_run():
    env.after(100, q_learning_update())
    env.mainloop()

def nn_run():
    env1.after(100, nn_update)
    env1.mainloop()

if __name__ == '__main__':
    env = Maze()
    env1 = Maze_dqn()
    table = QTable(actions=list(range(env.n_actions)))
    nn = DQN(env1.n_actions, env1.n_features,
             alpha=0.01,
             gamma=0.9,
             epsilon=0.9,
             replace_target_iter=200,
             memory_size=2000,
             epsilon_increment=0.1
             )
    nn_run()
    # q_learning_run()
