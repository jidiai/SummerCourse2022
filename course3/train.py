# -*- coding:utf-8  -*-
# Time  : 2022/8/19 下午5:25
# Author: Yahui Cui
import numpy as np
import os
import sys
import datetime

sys.path.append(os.path.dirname(__file__))
from env.chooseenv import make
from course3.examples.ddpg.submission import agent as ddpg_agent
from course3.examples.ddpg.submission import my_controller
from course3.examples.ddpg.submission import replay_buffer


def main():
    num_episodes = 200
    minimal_size = 1000
    batch_size = 64

    now = datetime.datetime.now()
    model_path = os.path.join(os.path.dirname(__file__), 'examples', 'ddpg', 'trained_model',
                              now.strftime("%Y-%m-%d-%H-%M-%S"))

    env_name = 'classic_Pendulum-v0'
    env = make(env_name)
    action_space = env.joint_action_space
    agent_id = 0

    return_list = []
    for i in range(10):

        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = my_controller(state[agent_id], action_space, True)
                next_state, reward, done, _, _ = env.step([action])
                replay_buffer.add(state[agent_id]['obs'], action[0], reward[agent_id], next_state[agent_id]['obs'], done)
                state = next_state
                episode_return += reward[agent_id]
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    ddpg_agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                print('episode' + ':' +
                      '%d' % (num_episodes / 10 * i + i_episode + 1) +
                      ' return' + ':''%.3f' % np.mean(return_list[-10:]))

        ddpg_agent.save(model_path, num_episodes / 10 * (i + 1))


if __name__ == '__main__':
    main()
