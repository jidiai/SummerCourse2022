import random

from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import time
import pygame
import sys
import os


from olympics_engine.generator import create_scenario
from pathlib import Path
current_path = str(Path(__file__).resolve().parent)
maps_path = os.path.join(current_path, "running_competition_maps/maps.json")



class Running_competition(OlympicsBase):
    def __init__(self, meta_map, map_id = None, seed = None, vis = None, vis_clear=None, agent1_color = 'purple', agent2_color = 'green'):
        # self.minimap_mode = map['obs_cfg'].get('minimap', False)

        Gamemap, map_index = Running_competition.choose_a_map(idx = map_id)        #fixme(yan): penatration in some maps, need to check engine, vis
        if vis is not None:
            for a in Gamemap['agents']:
                a.visibility = vis
                a.visibility_clear = vis_clear
                if a.color == 'purple':
                    a.color = agent1_color
                    a.original_color = agent1_color
                elif a.color == 'green':
                    a.color = agent2_color
                    a.original_color = agent2_color


        self.meta_map = meta_map
        self.map_index = map_index

        super(Running_competition, self).__init__(Gamemap, seed)

        self.game_name = 'running-competition'

        self.original_tau = meta_map['env_cfg']['tau']
        self.original_gamma = meta_map['env_cfg']['gamma']
        self.wall_restitution = meta_map['env_cfg']['wall_restitution']
        self.circle_restitution = meta_map['env_cfg']['circle_restitution']
        self.max_step = meta_map['env_cfg']['max_step']
        self.energy_recover_rate = meta_map['env_cfg']['energy_recover_rate']
        self.speed_cap = meta_map['env_cfg']['speed_cap']
        self.faster = meta_map['env_cfg']['faster']

        self.tau = self.original_tau*self.faster
        self.gamma = 1-(1-self.original_gamma)*self.faster

        # self.gamma = 1  # v衰减系数
        # self.restitution = 0.5
        # self.print_log = False
        # self.print_log2 = False
        # self.tau = 0.1
        #
        # self.speed_cap =  100
        #
        # self.draw_obs = True
        # self.show_traj = True

    @staticmethod
    def reset_map(meta_map, map_id, vis=None, vis_clear=None, agent1_color = 'purple', agent2_color = 'green'):
        return Running_competition(meta_map, map_id, vis=vis, vis_clear = vis_clear, agent1_color=agent1_color, agent2_color=agent2_color)

    @staticmethod
    def choose_a_map(idx=None):
        if idx is None:
            idx = random.randint(1,4)
        MapStats = create_scenario("map"+str(idx), file_path=  maps_path)
        return MapStats, idx

    def check_overlap(self):
        #todo
        pass

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]


        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 1.

        return agent_reward

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False



    def step(self, actions_list):

        previous_pos = self.agent_pos

        time1 = time.time()
        self.stepPhysics(actions_list, self.step_cnt)
        time2 = time.time()
        #print('stepPhysics time = ', time2 - time1)
        self.speed_limit()

        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward = self.get_reward()
        done = self.is_terminal()

        time3 = time.time()
        obs_next = self.get_obs()
        time4 = time.time()
        #print('render time = ', time4-time3)
        # obs_next = 1
        #self.check_overlap()
        self.change_inner_state()

        return obs_next, step_reward, done, ''

    def check_win(self):
        if self.agent_list[0].finished and not (self.agent_list[1].finished):
            return '0'
        elif not(self.agent_list[0].finished) and self.agent_list[1].finished:
            return '1'
        else:
            return '-1'


    def render(self, info=None):


        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode=True

        self.viewer.draw_background()
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)

        if self.draw_obs:
            if len(self.obs_list) > 0:
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=10, gap = 100)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)


        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()



if __name__ == '__main__':
    running = Running_competition()
    map = running.choose_a_map()
    print(map)