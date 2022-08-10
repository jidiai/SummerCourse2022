from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import pygame
import sys
import random
import os
from pathlib import Path
CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
import math


def point2point(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class wrestling(OlympicsBase):
    def __init__(self, map):
        self.minimap_mode = map['obs_cfg']['minimap']

        super(wrestling, self).__init__(map)

        self.game_name = 'wrestling'
        self.agent1_color = self.agent_list[0].color
        self.agent2_color = self.agent_list[1].color
        self.bound_color = 'green'


        self.gamma = map['env_cfg']['gamma']
        self.wall_restitution = map['env_cfg']['wall_restitution']
        self.circle_restitution = map['env_cfg']['circle_restitution']
        self.tau = map['env_cfg']['tau']
        self.speed_cap = map['env_cfg']['speed_cap']
        self.max_step = map['env_cfg']['max_step']
        self.print_log = False

        self.draw_obs = True
        self.show_traj = False
        self.beauty_render = False


    def check_overlap(self):
        pass


    def reset(self):
        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False


        init_obs = self.get_obs()
        if self.minimap_mode:
            self._build_minimap()

        output_init_obs = self._build_from_raw_obs(init_obs)
        return output_init_obs


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

        self.speed_limit()

        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward = self.get_reward()
        obs_next = self.get_obs()
        # obs_next = 1
        done = self.is_terminal()
        self.change_inner_state()

        if self.minimap_mode:
            self._build_minimap()

        output_obs_next = self._build_from_raw_obs(obs_next)

        return output_obs_next, step_reward, done, ''

    def _build_from_raw_obs(self, obs):
        if self.minimap_mode:
            image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)
            return [{"agent_obs": obs[0], "minimap":image, "id":"team_0"},
                    {"agent_obs": obs[1], "minimap": image, "id":"team_1"}]
        else:
            return [{"agent_obs":obs[0], "id":"team_0"}, {"agent_obs": obs[1], "id":"team_1"}]

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



    def cross_detect(self, previous_pos, new_pos):

        #case one: arc intersect with the agent
        #check radian first
        finals = []
        for object_idx in range(len(self.map['objects'])):
            object = self.map['objects'][object_idx]
            if object.can_pass() and object.color == self.bound_color:
                #arc_pos = object.init_pos
                finals.append(object)

        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            agent_pre_pos, agent_new_pos = previous_pos[agent_idx], new_pos[agent_idx]

            for final in finals:
                center = (final.init_pos[0] + 0.5*final.init_pos[2], final.init_pos[1]+0.5*final.init_pos[3])
                arc_r = final.init_pos[2]/2

                if final.check_radian(agent_new_pos, [0,0], 0):
                    l = point2point(agent_new_pos, center)
                    if abs(l - arc_r) <= agent.r:
                        # agent.color = 'blue'
                        agent.finished = True
                        agent.alive = False



        #case two: the agent cross the arc, inner  to outer or outer to inner


    def get_reward(self):

        agent1_finished = self.agent_list[0].finished
        agent2_finished = self.agent_list[1].finished
        if agent1_finished and agent2_finished:
            return [0., 0]
        elif agent1_finished and not agent2_finished:
            return [0., 1]
        elif not agent1_finished and agent2_finished:
            return [1., 0]
        else:
            return [0,0]

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False

    def check_win(self):
        if self.agent_list[0].finished and not (self.agent_list[1].finished):
            return '1'
        elif not (self.agent_list[0].finished) and self.agent_list[1].finished:
            return '0'
        else:
            return '-1'

    def render(self, info=None):

        if self.minimap_mode:
            pass
        else:

            if not self.display_mode:
                self.viewer.set_mode()
                self.display_mode = True

                if self.beauty_render:
                    self._load_image()
            self.viewer.draw_background(color_code=(108, 180, 143) if self.beauty_render else (255, 255, 255))
            if self.beauty_render:
                self._draw_playground()
                self._draw_energy(self.agent_list)

            for w in self.map['objects']:
                self.viewer.draw_map(w)

            if self.beauty_render:
                self._draw_image(self.agent_pos, self.agent_list, self.agent_theta, self.obs_boundary)
            else:
                self.viewer.draw_ball(self.agent_pos, self.agent_list)
                if self.draw_obs:
                    self.viewer.draw_obs(self.obs_boundary, self.agent_list)

        if self.draw_obs:
            if len(self.obs_list) > 0:
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=470, upmost_y=10, gap=130,
                                      energy_width=0 if self.beauty_render else 5)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        # self.viewer.draw_map()

        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)

        for event in pygame.event.get():
            # 如果单击关闭窗口，则退出
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
        # self.viewer.background.fill((255, 255, 255))

    def _load_image(self):
        self.playground_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/wrestling/playground.png"))
        r = 440
        self.playground_image = pygame.transform.scale(self.playground_image, size = (r*0.96899, r))

        self.player_1_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/wrestling/player1.png"))
        self.player_2_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/wrestling/player2.png"))

        self.player_1_view_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/wrestling/sight1.png")).convert_alpha()
        self.player_2_view_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/wrestling/sight2.png")).convert_alpha()

        self.wood_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/board.png")).convert_alpha()
        self.wood_image1 = pygame.transform.scale(self.wood_image, size = (260,170))

        self.red_energy_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-red.png")).convert_alpha()
        red_energy_size = self.red_energy_image.get_size()
        self.red_energy_image = pygame.transform.scale(self.red_energy_image, size = (110,red_energy_size[1]*110/red_energy_size[0]))

        self.blue_energy_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-blue.png")).convert_alpha()
        blue_energy_size = self.blue_energy_image.get_size()
        self.blue_energy_image = pygame.transform.scale(self.blue_energy_image, size = (110, blue_energy_size[1]*110/blue_energy_size[0]))

        self.red_energy_bar_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-red-bar.png")).convert_alpha()
        red_energy_bar_size = self.red_energy_bar_image.get_size()
        self.red_energy_bar_image = pygame.transform.scale(self.red_energy_bar_image, size=(85, 10))

        self.blue_energy_bar_image = pygame.image.load(os.path.join(CURRENT_PATH, "assets/energy-blue-bar.png")).convert_alpha()
        blue_energy_bar_size = self.blue_energy_bar_image.get_size()
        self.blue_energy_bar_image = pygame.transform.scale(self.blue_energy_bar_image, size=(85, 10))

    def _draw_playground(self):
        loc = (91,133)
        self.viewer.background.blit(self.playground_image, loc)
        self.viewer.background.blit(self.wood_image1, (440, 0))
        self.viewer.background.blit(self.red_energy_image, (450, 130))
        self.viewer.background.blit(self.blue_energy_image, (580, 130))


    def _draw_energy(self, agent_list):

        # red_energy_bar_size = self.red_energy_bar_image.get_size()
        # blue_energy_bar_size = self.blue_energy_bar_image.get_size()
        # red_energy_bar = pygame.transform.scale(self.red_energy_bar_image, size=(85, 10))
        # blue_energy_bar = pygame.transform.scale(self.blue_energy_bar_image, size=(85, 10))

        start_pos = [473, 136]
        # end_pos = [450+100*remain_energy, 130]
        image = self.red_energy_bar_image
        for agent_idx in range(len(agent_list)):
            if agent_list[agent_idx].type == 'ball':
                continue

            remain_energy = agent_list[agent_idx].energy/agent_list[agent_idx].energy_cap


            self.viewer.background.blit(image, start_pos, [0, 0, 85*remain_energy, 10])

            start_pos[0] += 130
            image = self.blue_energy_bar_image


    def _draw_image(self, pos_list, agent_list, direction_list, view_list):
        assert len(pos_list) == len(agent_list)
        for i in range(len(pos_list)):
            agent = self.agent_list[i]

            t = pos_list[i]
            r = agent_list[i].r
            color = agent_list[i].color
            theta = direction_list[i][0]
            vis = agent_list[i].visibility
            view_back = self.VIEW_BACK*vis if vis is not None else 0
            if agent.type == 'agent':
                if color == self.agent1_color:
                    player_image_size = self.player_1_image.get_size()
                    image= pygame.transform.scale(self.player_1_image, size = (r*2, player_image_size[1]*(r*2)/player_image_size[0]))
                    loc = (t[0]-r ,t[1]-r)

                    view_image = pygame.transform.scale(self.player_1_view_image, size = (vis, vis))
                    rotate_view_image = pygame.transform.rotate(view_image, -theta)

                    new_view_center = [t[0]+(vis/2-view_back)*math.cos(theta*math.pi/180), t[1]+(vis/2-view_back)*math.sin(theta*math.pi/180)]
                    new_view_rect = rotate_view_image.get_rect(center=new_view_center)
                    self.viewer.background.blit(rotate_view_image, new_view_rect)

                    #view player image
                    player_image_view = pygame.transform.rotate(image, 90)
                    self.viewer.background.blit(player_image_view, (480, 90))


                elif color == self.agent2_color:
                    player_image_size = self.player_2_image.get_size()
                    image= pygame.transform.scale(self.player_2_image, size = (r*2, player_image_size[1]*(r*2)/player_image_size[0]))
                    loc = (t[0]-r ,t[1]-r)

                    view_image = pygame.transform.scale(self.player_2_view_image, size = (vis, vis))
                    rotate_view_image = pygame.transform.rotate(view_image, -theta)

                    new_view_center = [t[0]+(vis/2-view_back)*math.cos(theta*math.pi/180), t[1]+(vis/2-view_back)*math.sin(theta*math.pi/180)]
                    new_view_rect = rotate_view_image.get_rect(center=new_view_center)
                    self.viewer.background.blit(rotate_view_image, new_view_rect)

                    player_image_view = pygame.transform.rotate(image, 90)
                    self.viewer.background.blit(player_image_view, (610, 90))

                    # self.viewer.background.blit(image_green, loc)

            rotate_image = pygame.transform.rotate(image, -theta)

            new_rect = rotate_image.get_rect(center=image.get_rect(center = t).center)


            self.viewer.background.blit(rotate_image, new_rect)


