from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
from olympics_engine.objects import Ball, Agent
from pathlib import Path
CURRENT_PATH = str(Path(__file__).resolve().parent.parent)

import numpy as np
import math
import pygame
import sys
import os

from typing import List, Dict

# color 宏
COLORS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'grey':  [176,196,222],
    'purple': [160, 32, 240],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'light green': [204, 255, 229],
    'sky blue': [0,191,255]
}

COLOR_TO_IDX = {
    'red': 7,
    'green': 1,
    'sky blue': 2,
    'yellow': 3,
    'grey': 4,
    'purple': 5,
    'black': 6,
    'light green': 0,
    'blue':8

}

IDX_TO_COLOR = {
    0: 'light green',
    1: 'green',
    2: 'sky blue',
    3: 'yellow',
    4: 'grey',
    5: 'purple',
    6: 'black',
    7: 'red',
    8: 'blue'
}

grid_node_width = 2     #for view drawing
grid_node_height = 2


from olympics_engine.tools.func import closest_point, distance_to_line




class curling(OlympicsBase):
    def __init__(self, map):
        self.minimap_mode = map['obs_cfg']['minimap']
        self.original_tau = map['env_cfg']['tau']
        self.original_gamma = map['env_cfg']['gamma']

        super(curling, self).__init__(map)

        self.game_name = 'curling'

        self.tau = self.original_tau
        self.faster = map['env_cfg']['faster']

        self.wall_restitution = map['env_cfg']['wall_restitution']
        self.circle_restitution = map['env_cfg']['circle_restitution']
        self.round_max_step = map['env_cfg']['round_max_step']
        self.max_n = map['env_cfg']['max_n']
        self.field_gamma = map['env_cfg']["field_gamma"]

        self.print_log = False
        self.draw_obs = True
        self.show_traj = False
        self.start_pos = [300,150]
        self.start_init_obs = 90

        self.vis=200
        self.vis_clear = 5

        self.purple_rock = pygame.image.load(os.path.join(CURRENT_PATH, "assets/purple rock.png"))
        self.green_rock = pygame.image.load(os.path.join(CURRENT_PATH,"assets/green rock.png"))
        self.curling_ground = pygame.image.load(os.path.join(CURRENT_PATH, "assets/curling ground.png"))
        self.crown_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/crown.png"))
        # self.curling_ground.set_alpha(150)

    def reset(self):
        self.purple_rock = pygame.image.load(os.path.join(CURRENT_PATH, "assets/purple rock.png"))
        self.green_rock = pygame.image.load(os.path.join(CURRENT_PATH,"assets/green rock.png"))
        self.curling_ground = pygame.image.load(os.path.join(CURRENT_PATH, "assets/curling ground.png"))
        self.crown_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/crown.png"))

        self.release = False
        self.gamma = self.original_gamma
        self.tau = self.original_tau

        self.num_purple = 1
        self.num_green = 0
        self.temp_winner = -1
        self.round_step = 0

        self.agent_num = 0
        self.agent_list = []
        self.agent_init_pos = []
        self.agent_pos = []
        self.agent_previous_pos = []
        self.agent_v = []
        self.agent_accel = []
        self.agent_theta = []

        self.obs_boundary_init = list()
        self.obs_boundary = self.obs_boundary_init

        #self.check_valid_map()
        self.generate_map(self.map)
        self.merge_map()

        self.init_state()
        self.step_cnt = 0
        self.done = False
        self.release = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False
        self.view_terminal = False

        self.current_team = 0
        obs = self.get_obs()

        if self.minimap_mode:
            self._build_minimap()
        output_init_obs = self._build_from_raw_obs(obs)
        return output_init_obs

        # return [obs, np.zeros_like(obs)-1]

    def _reset_round(self):
        self.current_team = 1-self.current_team
        #convert last agent to ball
        if len(self.agent_list) != 0:
            last_agent = self.agent_list[-1]
            last_ball = Ball(mass = last_agent.mass, r = last_agent.r, position = self.agent_pos[-1],
                             color = last_agent.color)
            last_ball.alive = False
            self.agent_list[-1] = last_ball

        #add new agent
        if self.current_team == 0:
            #team purple
            new_agent_color = 'purple'
            self.num_purple += 1

        elif self.current_team == 1:
            new_agent_color = 'green'
            self.num_green += 1

        else:
            raise NotImplementedError

        new_agent = Agent(mass = 1, r= 15, position = self.start_pos, color = new_agent_color,
                          vis = self.vis, vis_clear = self.vis_clear)

        self.agent_list.append(new_agent)
        self.agent_init_pos[-1] = self.start_pos
        new_boundary = self.get_obs_boundaray(self.start_pos, 15, self.vis)
        self.obs_boundary_init.append(new_boundary)
        self.agent_num += 1

        self.agent_pos.append(self.agent_init_pos[-1])
        self.agent_v.append([0,0])
        self.agent_accel.append([0,0])
        init_obs = self.start_init_obs
        self.agent_theta.append([init_obs])
        self.agent_record.append([self.agent_init_pos[-1]])

        self.release = False
        self.gamma = self.original_gamma
        self.tau = self.original_tau

        self.round_step = 0

        return self.get_obs()




    def cross_detect(self):
        """
        check whether the agent has reach the cross(final) line
        :return:
        """
        for agent_idx in range(self.agent_num):

            agent = self.agent_list[agent_idx]
            if agent.type != 'agent':
                continue

            for object_idx in range(len(self.map['objects'])):
                object = self.map['objects'][object_idx]

                if not object.can_pass():
                    continue
                else:
                    #print('object = ', object.type)
                    if object.color == 'red' and object.type=='cross' and \
                            object.check_cross(self.agent_pos[agent_idx], agent.r):
                        # print('agent type = ', agent.type)
                        agent.alive = False
                        #agent.color = 'red'
                        self.gamma = 1-(1-self.field_gamma)*self.faster
                        self.tau *= self.faster
                        self.release = True
                        self.round_countdown = self.round_max_step-self.round_step
                    # if the ball hasnot pass the cross, the relase will be True again in the new round
    def check_action(self, action_list):
        action = []
        for agent_idx in range(len(self.agent_list)):
            if self.agent_list[agent_idx].type == 'agent':
                action.append(action_list[0])
                _ = action_list.pop(0)
            else:
                action.append(None)

        return action



    def step(self, actions_list):

        actions_list = [actions_list[self.current_team]]

        #previous_pos = self.agent_pos
        action_list = self.check_action(actions_list)
        if self.release:
            input_action = [None for _ in range(len(self.agent_list))]       #if jump, stop actions
        else:
            input_action = action_list

        self.stepPhysics(input_action, self.step_cnt)

        if not self.release:
            self.cross_detect()
        self.step_cnt += 1
        self.round_step += 1

        obs_next = self.get_obs()


        done = self.is_terminal()

        if not done:
            round_end, end_info = self._round_terminal()
            if round_end:

                if end_info is not None:
                    #clean the last agent
                    del self.agent_list[-1]
                    del self.agent_pos[-1]
                    del self.agent_v[-1]
                    del self.agent_theta[-1]
                    del self.agent_accel[-1]
                    self.agent_num -= 1

                self.temp_winner, min_d = self.current_winner()
                #step_reward = [1,0.] if self.temp_winner == 0 else [0., 1]          #score for each round
                if self.temp_winner == -1:
                    step_reward=[0., 0.]
                elif self.temp_winner == 0:
                    step_reward=[1, 0.]
                elif self.temp_winner == 1:
                    step_reward=[0., 1]
                else:
                    raise NotImplementedError


                obs_next = self._reset_round()

            else:
                step_reward = [0., 0.]

        else:
            self.final_winner, min_d = self.current_winner()
            step_reward = [100., 0] if self.final_winner == 0 else [0., 100]
            self.view_terminal = True

        # if self.current_team == 0:
        #     obs_next = [obs_next, np.zeros_like(obs_next)-1]
        # else:
        #     obs_next = [np.zeros_like(obs_next)-1, obs_next]
        if self.minimap_mode:
            self._build_minimap()
        output_obs_next = self._build_from_raw_obs(obs_next)

        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return output_obs_next, step_reward, done, ''

    def get_obs_encode(self):
        obs = self.get_obs()
        if self.current_team == 0:
            return [obs, np.zeros_like(obs)]
        else:
            return [np.zeros_like(obs), obs]


    def _build_from_raw_obs(self, obs):
        if self.minimap_mode:
            image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)

        if self.current_team == 0:
            return [{'agent_obs': obs[0], "minimap": image if self.minimap_mode else None, "id": "team_0"},
                    {'agent_obs': np.zeros_like(obs[0]) - 1, "minimap": None, "id": "team_1"}]
        elif self.current_team == 1:
            return [{"agent_obs": np.zeros_like(obs[1]) - 1, "minimap": None, "id": "team_0"},
                    {"agent_obs": obs[1], "minimap": image if self.minimap_mode else None, "id": "team_1"}]

    def _build_minimap(self):
        #need to render first
        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode = True

        self.viewer.draw_background()

        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)


    def get_reward(self):

        center = [300, 500]
        pos = self.agent_pos[0]
        distance = math.sqrt((pos[0]-center[0])**2 + (pos[1]-center[1])**2)
        return [distance]

    def is_terminal(self):

        # if self.step_cnt >= self.max_step:
        #     return True

        if (self.num_green + self.num_purple == self.max_n*2):
            if not self.release and self.round_step > self.round_max_step:
                return True

            if self.release:
                L = []
                for agent_idx in range(self.agent_num):
                    if (self.agent_v[agent_idx][0] ** 2 + self.agent_v[agent_idx][1] ** 2) < 1e-1:
                        L.append(True)
                    else:
                        L.append(False)
                return all(L)
        else:
            return False

        # for agent_idx in range(self.agent_num):
        #     if self.agent_list[agent_idx].color == 'red' and (
        #             self.agent_v[agent_idx][0] ** 2 + self.agent_v[agent_idx][1] ** 2) < 1e-5:
        #         return True

    def _round_terminal(self):

        if self.round_step > self.round_max_step and not self.release:      #after maximum round step the agent has not released yet
            return True, -1

        #agent_idx = -1
        L = []
        for agent_idx in range(self.agent_num):
            if (not self.agent_list[agent_idx].alive) and (self.agent_v[agent_idx][0] ** 2 +
                                                        self.agent_v[agent_idx][1] ** 2) < 1e-1:
                L.append(True)
            else:
                L.append(False)

        return all(L), None

    def current_winner(self):

        center = [300, 500]
        min_dist = 1e4
        win_team = -1
        for i, agent in enumerate(self.agent_list):
            pos = self.agent_pos[i]
            distance = math.sqrt((pos[0]-center[0])**2 + (pos[1]-center[1])**2)
            if distance < min_dist:
                win_team = 0 if agent.color == 'purple' else 1
                min_dist = distance

        return win_team, min_dist



    def render(self, info=None):

        if self.minimap_mode:
            ground_image = pygame.transform.scale(self.curling_ground, size=(200, 200))
            self.viewer.background.blit(ground_image, (200, 400))

            if self.draw_obs:
                self.viewer.draw_obs(self.obs_boundary, self.agent_list)
            # 先画map; ball在map之上
            for w in self.map['objects']:
                if w.type=='arc':
                    continue
                self.viewer.draw_map(w)

            self._draw_curling_rock(self.agent_pos, self.agent_list)

        else:

            if not self.display_mode:
                self.viewer.set_mode()
                self.display_mode=True

            self.viewer.draw_background()

            ground_image = pygame.transform.scale(self.curling_ground, size=(200, 200))
            self.viewer.background.blit(ground_image, (200, 400))

            if self.draw_obs:
                self.viewer.draw_obs(self.obs_boundary, self.agent_list)


            # 先画map; ball在map之上
            for w in self.map['objects']:
                if w.type=='arc':
                    continue
                self.viewer.draw_map(w)

            self._draw_curling_rock(self.agent_pos, self.agent_list)


        # self.viewer.draw_ball(self.agent_pos, self.agent_list)
        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)
        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        #self.viewer.draw_map()




        if self.draw_obs:
            if self.current_team == 0:
                # self.viewer.draw_view(self.obs_list, [self.agent_list[-1]])
                # self.viewer.draw_curling_view(self.purple_rock,self.green_rock,self.obs_list, [self.agent_list[-1]])
                self._draw_curling_view(self.obs_list, self.agent_list)
            else:
                # self.viewer.draw_view([None, self.obs_list[0]], [None, self.agent_list[-1]])
                # self.viewer.draw_curling_view(self.purple_rock, self.green_rock, [None, self.obs_list[0]], [None, self.agent_list[-1]])
                self._draw_curling_view(self.obs_list, self.agent_list)


            debug('Agent 0', x=570, y=110, c='purple')
            debug("No. throws left: ", x=470, y=140)
            debug("{}".format(self.max_n - self.num_purple), x = 590, y=140, c='purple')
            debug('Agent 1', x=640, y=110, c='green')
            debug("{}".format(self.max_n - self.num_green), x=660, y = 140, c='green')
            debug("Current winner:", x=470, y=170)

            if self.view_terminal:
                crown_size=(50,50)
            else:
                crown_size=(30,30)
            crown_image = pygame.transform.scale(self.crown_image, size=crown_size)
            if self.temp_winner == 0:
                self.viewer.background.blit(crown_image, (570, 150) if self.view_terminal else (580, 160))
            elif self.temp_winner == 1:
                self.viewer.background.blit(crown_image, (640, 150) if self.view_terminal else (650, 160))
            else:
                pass


            pygame.draw.line(self.viewer.background, start_pos=[470, 130], end_pos=[690, 130], color=[0,0,0])
            pygame.draw.line(self.viewer.background, start_pos=[565, 100], end_pos=[565,190], color=[0,0,0])
            pygame.draw.line(self.viewer.background, start_pos=[630, 100], end_pos=[630,190], color=[0,0,0])
            pygame.draw.line(self.viewer.background, start_pos=[470, 160], end_pos=[690, 160], color=[0,0,0])


        #draw energy bar
        #debug('agent remaining energy = {}'.format([i.energy for i in self.agent_list]), x=100)
        # self.viewer.draw_energy_bar(self.agent_list)


        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)

        if not self.release:
            countdown = self.round_max_step-self.round_step
        else:
            countdown = self.round_countdown

        debug("Countdown:", x=100)
        debug("{}".format(countdown), x=170, c="red")
        debug("Current winner:", x=200)

        if self.temp_winner == -1:
            debug("None", x = 300)
        elif self.temp_winner == 0:
            debug("Purple", x=300, c='purple')
        elif self.temp_winner == 1:
            debug("Green", x=300, c='green')


        if info is not None:
            debug(info, x=100)


        for event in pygame.event.get():
            # 如果单击关闭窗口，则退出
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
        #self.viewer.background.fill((255, 255, 255))

    def _draw_curling_rock(self, pos_list, agent_list):

        assert len(pos_list) == len(agent_list)
        for i in range(len(pos_list)):
            t = pos_list[i]
            r = agent_list[i].r
            color = agent_list[i].color

            if color == 'purple':
                image_purple = pygame.transform.scale(self.purple_rock, size=(r * 2, r * 2))
                loc = (t[0] - r, t[1] - r)
                self.viewer.background.blit(image_purple, loc)
            elif color == 'green':
                image_green = pygame.transform.scale(self.green_rock, size=(r * 2, r * 2))
                loc = (t[0] - r, t[1] - r)
                self.viewer.background.blit(image_green, loc)
            else:
                raise NotImplementedError

    def _draw_curling_view(self, obs: List, agent_list: List):       #obs: [2, 100, 100] list

        #draw agent 1, [50, 50], [50+width, 50], [50, 50+height], [50+width, 50+height]
        # coord = [580 + 70 * i for i in range(len(obs))]
        matrix = obs[-1]
        agent = agent_list[-1]
        color = agent.color
        r=  agent.r
        coord = 570 if agent.color == 'purple' else 570+70
        count = int(color == 'green')

        obs_weight, obs_height = matrix.shape[0], matrix.shape[1]
        y = 40-obs_weight
        for row in matrix:
            x = coord - obs_height / 2
            for item in row:
                pygame.draw.rect(self.viewer.background, COLORS[IDX_TO_COLOR[int(item)]],
                                 [x, y, grid_node_width, grid_node_height])
                x += grid_node_width
            y += grid_node_height

        if color == 'purple':
            image_purple = pygame.transform.scale(self.purple_rock, size=(r * 2, r * 2))
            loc = [coord + 20 - r, 78 ]
            self.viewer.background.blit(image_purple, loc)
        elif color == 'green':
            image_green = pygame.transform.scale(self.green_rock, size=(r * 2, r * 2))
            loc = [coord + 20 - r, 78 ]
            self.viewer.background.blit(image_green, loc)
        else:
            raise NotImplementedError

        pygame.draw.lines(self.viewer.background, points=[[550 + 68 * count, 0], [550 + 70 * count, 78],
                                                          [550 + 78 + 70 * count, 78],
                                                          [550 + 78 + 70 * count, 0]], closed=True,
                          color=COLORS[color], width=2)
        #
        # for agent_idx in range(len(obs)):
        #     matrix = obs[agent_idx]
        #     if matrix is None:
        #         continue
        #
        #     obs_weight, obs_height = matrix.shape[0], matrix.shape[1]
        #     y = 40 - obs_height
        #     for row in matrix:
        #         x = coord[agent_idx]- obs_height/2
        #         for item in row:
        #             pygame.draw.rect(self.viewer.background, COLORS[IDX_TO_COLOR[int(item)]], [x,y,grid_node_width, grid_node_height])
        #             x+= grid_node_width
        #         y += grid_node_height
        #
        #     color = agent_list[agent_idx].color
        #     r = agent_list[agent_idx].r
        #
        #     if color == 'purple':
        #         image_purple = pygame.transform.scale(self.purple_rock, size=(r*2, r*2))
        #         loc = [coord[agent_idx]+15-r, 70 + agent_list[agent_idx].r-r]
        #         self.viewer.background.blit(image_purple, loc)
        #     elif color == 'green':
        #         image_green = pygame.transform.scale(self.green_rock, size=(r*2, r*2))
        #         loc = [coord[agent_idx]+15-r, 70 + agent_list[agent_idx].r-r]
        #         self.viewer.background.blit(image_green, loc)
        #     else:
        #         raise NotImplementedError
        #
        #     #
        #     # pygame.draw.circle(self.background, COLORS[agent_list[agent_idx].color], [coord[agent_idx]+10, 55 + agent_list[agent_idx].r],
        #     #                    agent_list[agent_idx].r, width=0)
        #     # pygame.draw.circle(self.background, COLORS["black"], [coord[agent_idx]+10, 55 + agent_list[agent_idx].r], 2,
        #     #                    width=0)
        #
        #     pygame.draw.lines(self.viewer.background, points =[[563+70*agent_idx,10],[563+70*agent_idx, 70], [565+60+70*agent_idx,70], [565+60+70*agent_idx, 10]], closed=True,
        #                       color = COLORS[agent_list[agent_idx].color], width=2)



