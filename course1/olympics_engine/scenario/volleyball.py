from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import debug
import pygame
import sys
import math


class volleyball(OlympicsBase):
    def __init__(self, map):
        super(volleyball, self).__init__(map)

        self.gamma = 1  # v衰减系数
        self.restitution = 0.7
        self.print_log = False
        self.tau = 0.1

        self.draw_obs = True
        self.show_traj = True

        self.g = 60
        self.agent_original_accel = [[0,0], [0,0]]

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

    def actions_to_accel(self, actions_list):
        self.agent_original_accel = [[] for _ in range(self.agent_num)]
        a_container = [[] for _ in range(self.agent_num)]
        for agent_idx in range(self.agent_num):
            action = actions_list[agent_idx]
            if action is None:
                accel = [0, self.agent_list[agent_idx].mass*self.g]
                self.agent_original_accel[agent_idx] = [0,0]

            else:
                if self.agent_list[agent_idx].is_fatigue:       #if agent is out of energy, no driving force applies
                    accel = [0,self.agent_list[agent_idx].mass*self.g]
                else:
                    mass = self.agent_list[agent_idx].mass

                    assert self.action_f[0] <= action[0] <= self.action_f[1], print('Continuous driving force needs '
                                                                                    'to be within the range [-100,200]')
                    force = action[0] / mass
                    assert self.action_theta[0] <= action[1] <= self.action_theta[1], print(
                        'Continuous turing angle needs to be within the range [-30deg, 30deg]')
                    theta = action[1]

                    theta_old = self.agent_theta[agent_idx][0]
                    theta_new = theta_old + theta
                    self.agent_theta[agent_idx][0] = theta_new

                    accel_x = force * math.cos(theta_new / 180 * math.pi)
                    accel_y = force * math.sin(theta_new / 180 * math.pi)
                    accel = [accel_x, accel_y + mass*self.g ]
                self.agent_original_accel[agent_idx] = [accel_x, accel_y]

            a_container[agent_idx] = accel
        return a_container



    def step(self, actions_list):
        previous_pos = self.agent_pos

        actions_list = self.check_action(actions_list)

        self.stepPhysics(actions_list, self.step_cnt)

        #self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward = 1 #self.get_reward()
        obs_next = self.get_obs()
        # obs_next = 1
        done = False#self.is_terminal()

        #check overlapping
        #self.check_overlap()

        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return obs_next, step_reward, done, ''

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
            self.viewer.draw_view(self.obs_list, self.agent_list)

        #draw energy bar
        #debug('agent remaining energy = {}'.format([i.energy for i in self.agent_list]), x=100)
        self.viewer.draw_energy_bar(self.agent_list)
        debug('Agent 0', x=570, y=110)
        debug('Agent 1', x=640, y=110)
        if self.map_num is not None:
            debug('Map {}'.format(self.map_num), x=100)

        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)
        debug("Gravity", x = 100)
        pygame.draw.line(self.viewer.background, color=[0,0,0],start_pos=[160,10], end_pos=[160,30], width=4)
        pygame.draw.line(self.viewer.background, color=[0,0,0],start_pos=[160,30], end_pos=[155,25], width=4)
        pygame.draw.line(self.viewer.background, color=[0,0,0],start_pos=[160,30], end_pos=[165,25], width=4)


        for event in pygame.event.get():
            # 如果单击关闭窗口，则退出
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
        #self.viewer.background.fill((255, 255, 255))
