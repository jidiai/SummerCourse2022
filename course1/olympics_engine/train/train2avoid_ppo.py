# here lies an example of how to train an RL agent on individual subgame


import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)


from collections import deque, namedtuple
import random

from olympics_engine.generator import create_scenario
from env_wrapper.chooseenv import make
from train.log_path import *
from train.algo.ppo import PPO
from train.algo.random import random_agent

# from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition
# from olympics_engine.agent import *

from olympics_engine.core import OlympicsBase
from olympics_engine.objects import *
from olympics_engine.viewer import Viewer, debug

import pygame
import time
import math



gamemap = {'objects':[], 'agents':[]}

gamemap['objects'].append(Wall(init_pos=[[50, 250], [650, 250]], length = None, color = 'black'))
gamemap['objects'].append(Wall(init_pos=[[50, 450], [650, 450]], length = None, color = 'black'))
gamemap['objects'].append(Wall(init_pos=[[50, 250], [50, 450]], length = None, color = 'black'))

gamemap['objects'].append(Cross(init_pos=[[650, 250], [650, 450]], length = None, color = 'red', width = 5))
gamemap['objects'].append(Cross(init_pos=[[200, 250], [200, 300]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[250, 450], [250, 400]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[300, 300], [300, 350]], length = None, color = 'green', width = 5))

gamemap['objects'].append(Cross(init_pos=[[450, 250], [450, 350]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[550, 450], [550, 350]], length = None, color = 'green', width = 5))




gamemap['agents'].append(Agent(position = [75,300], mass=1, r=15, color='light red', vis_clear=5, vis=200))
gamemap['view'] = {'width': 600, 'height':600, 'edge': 50, "init_obs": [0]}



def point2point(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class env_test(OlympicsBase):
    def __init__(self, map=gamemap, point_left_prob = 1):

        # if random.uniform(0,1) <= point_left_prob:
        #     self.final = 'left'
        #     gamemap['objects'].append(Cross(init_pos=[[330, 600], [370, 580]], length = None, color = 'grey'))
        #     gamemap['objects'].append(Cross(init_pos=[[330, 600], [370, 620]], length = None, color = 'grey'))
        # else:
        #     self.final = 'right'
        #     gamemap['objects'].append(Cross(init_pos=[[370, 600], [330, 580]], length = None, color = 'grey'))
        #     gamemap['objects'].append(Cross(init_pos=[[370, 600], [330, 620]], length = None, color = 'grey'))


        super(env_test, self).__init__(map)

        self.map = map
        self.gamma = 1  # v衰减系数
        self.wall_restitution = 0.5
        self.print_log = False
        self.tau = 0.1
        self.max_step = 300

        self.draw_obs = True
        self.show_traj = False

        self.cross_color = 'red'
        self.penalty_color = 'green'
        finals = []
        penalty = []
        for object_idx in range(len(self.map['objects'])):
            object = self.map['objects'][object_idx]
            if object.can_pass():
                if object.color == self.cross_color:
                    finals.append(object)
                elif object.color == self.penalty_color:
                    penalty.append(object)
        self.finals = finals
        self.penalty = penalty
        self.info = ''

    def check_overlap(self):
        pass

    def check_action(self, action_list):
        action = []
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'agent':
                action.append(action_list[0])
                _ = action_list.pop(0)
            else:
                action.append(None)

        return action

    def step(self, actions_list):

        previous_pos = self.agent_pos
        actions_list = self.check_action(actions_list)

        self.stepPhysics(actions_list, self.step_cnt)
        self.cross_detect(self.agent_pos)

        self.step_cnt += 1
        step_reward = self.get_reward()
        obs_next = self.get_obs()
        # obs_next = 1
        done = self.is_terminal()

        #check overlapping
        #self.check_overlap()

        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return obs_next, step_reward, done, self.info

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                if self.agent_list[agent_idx].color == self.cross_color:
                    agent_reward[agent_idx] = 10
                elif self.agent_list[agent_idx].color == self.penalty_color:
                    agent_reward[agent_idx] = -10
            else:
                if self.step_cnt >= self.max_step:
                    agent_reward[agent_idx] = -10


            current_pos_x = self.agent_pos[agent_idx][0]
            pos_r = -(600-current_pos_x)/650
            agent_reward[agent_idx] += (pos_r + 1)*0.05

        return agent_reward


    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False

    def cross_detect(self, new_pos):
        # finals = []
        # penalty = []
        # for object_idx in range(len(self.map['objects'])):
        #     object = self.map['objects'][object_idx]
        #     if object.can_pass():
        #         if object.color == self.cross_color:
        #             finals.append(object)
        #         elif object.color == self.penalty_color:
        #             penalty.append(object)

        self.info = ''
        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            agent_checked = False
            for final in self.finals:
                if final.check_cross(self.agent_pos[agent_idx], agent.r):
                    agent.color = self.cross_color
                    agent.finished = True  # when agent has crossed the finished line
                    agent.alive = False
                    agent_checked = True
                    self.info = 'finished'

            if not agent_checked:
                for pen in self.penalty:
                    if pen.check_cross(self.agent_pos[agent_idx], agent.r):
                        agent.color = self.penalty_color
                        agent.finished = True  # when agent has crossed the finished line
                        agent.alive = False


    def render(self, info=None):

        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode=True

        self.viewer.draw_background()
        # 先画map; ball在map之上
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)
        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)
        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        #self.viewer.draw_map()

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)
            self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=5)

        #draw energy bar
        #debug('agent remaining energy = {}'.format([i.energy for i in self.agent_list]), x=100)
        # self.viewer.draw_energy_bar(self.agent_list)


        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)


        for event in pygame.event.get():
            # 如果单击关闭窗口，则退出
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
        #self.viewer.background.fill((255, 255, 255))









parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="Learn2Avoid", type=str, help='running-competition/table-hockey/football/wrestling')
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=1500, type=int)
parser.add_argument('--episode_length', default=500, type=int)

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=900, type=int)


device = 'cpu'
RENDER = True
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


def main(args):
    num_agents = 1
    ctrl_agent_index = 0        #controlled agent index


    # if args.game_name == 'running-competition':
    #     map_id = random.randint(1,4)
    #     # map_id = 3
    #     Gamemap = create_scenario(args.game_name)
    #     env = Running_competition(meta_map=Gamemap,map_id=map_id, vis = 200, vis_clear=5, agent1_color = 'light red',
    #                               agent2_color = 'blue')
    #     env.max_step = 400
    # elif args.game_name == 'table-hockey':
    #     Gamemap = create_scenario(args.game_name)
    #     env = table_hockey(Gamemap)
    #     env.max_step = 400
    # elif args.game_name == 'football':
    #     Gamemap = create_scenario(args.game_name)
    #     env = football(Gamemap)
    #     env.max_step = 400
    # elif args.game_name == 'wrestling':
    #     Gamemap = create_scenario(args.game_name)
    #     env = wrestling(Gamemap)
    #     env.max_step = 400
    # else:
    #     raise NotImplementedError
    env = env_test()

    print(f'Playing game {args.game_name}')
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    print(f'Total agent number: {num_agents}')
    print(f'Agent control by the actor: {ctrl_agent_index}')


    # width = env.view_setting['width']+2*env.view_setting['edge']
    # height = env.view_setting['height']+2*env.view_setting['edge']
    # print(f'Game board width: {width}')
    # print(f'Game board height: {height}')

    obs_dim = 40*40
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on subgames {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args.algo, args.game_name)))
        save_config(args, log_dir)

    record_win = deque(maxlen=100)
    # record_win_op = deque(maxlen=100)

    if args.load_model:
        model = PPO()
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir,episode=args.load_episode)
    else:
        model = PPO(run_dir)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    # opponent_agent = random_agent()     #we use random opponent agent here

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        state = env.reset()
        if RENDER:
            env.render()
        # if isinstance(state[ctrl_agent_index], type({})):
        #     obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
            # obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
        # else:
        #     obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index].flatten(), env.agent_list[ctrl_agent_index].energy
        #     obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy

        obs_ctrl_agent = state[ctrl_agent_index].flatten()

        episode += 1
        step = 0
        Gt = 0

        while True:
            # action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
            # action_opponent = [0,0]  #here we assume the opponent is not moving in the demo

            action_ctrl_raw, action_prob= model.select_action(obs_ctrl_agent, False if args.load_model else True)
            #inference
            action_ctrl = actions_map[action_ctrl_raw]

            # action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]
            action = [action_ctrl]

            next_state, reward, done, info = env.step(action)

            # if isinstance(next_state[ctrl_agent_index], type({})):
            #     next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
            #     next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
            # else:
            #     next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
            #     next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy
            next_obs_ctrl_agent = next_state[ctrl_agent_index].flatten()


            step += 1

            # if not done:
                # post_reward = [-1., -1.]
            post_reward = reward
            # else:
                # if reward[0] != reward[1]:
                #     post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                # else:
                #     post_reward=[-1., -1.]


            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)

            # obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
            # obs_ctrl_agent, energy_ctrl_agent = np.array(next_obs_ctrl_agent).flatten(), next_energy_ctrl_agent
            obs_ctrl_agent = next_obs_ctrl_agent

            if RENDER:
                env.render()
            Gt += reward[ctrl_agent_index]

            if done:
                win_is = 1 if info == 'finished' else 0   #>reward[1-ctrl_agent_index] else 0
                # win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                # record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,'; Trained episode:', train_count,
                      "win rate = ", sum(record_win)/len(record_win))

                if not args.load_model:
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        # if win_is == 1:
                        model.update(episode)
                        train_count += 1
                        # else:
                        #     model.clear_buffer()

                    writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)





if __name__ == '__main__':
    args = parser.parse_args()
    main(args)