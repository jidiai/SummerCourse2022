import numpy as np
from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
from olympics_engine.objects import Agent
import pygame
import sys
import math
import copy
import random

def point2point(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class billiard_joint(OlympicsBase):
    def __init__(self, map):
        self.minimap_mode = map['obs_cfg']['minimap']

        super(billiard_joint, self).__init__(map)

        self.tau = map['env_cfg']['tau']
        self.gamma = map['env_cfg']['gamma']
        self.wall_restitution = map['env_cfg']['wall_restitution']
        self.circle_restitution = map['env_cfg']['circle_restitution']
        self.max_step = map['env_cfg']['max_step']
        self.max_n_hit = map['env_cfg']['max_n_hit']
        self.white_penalty = map['env_cfg']['white_penalty']
        self.pot_reward = map['env_cfg']['pot_reward']

        self.print_log = False

        self.draw_obs = True
        self.show_traj = False

        self.dead_agent_list = [[],[]]
        self.original_num_ball = len(self.agent_list)
        self.white_ball_in = [False, False]

        # self.purple_init_pos =  [[50, 200], [200, 350]]   #[ [x_min, xmax], [ymin, ymax]]     #[[100,325], [100,425]]
        # self.green_init_pos = [[50,200], [400,550]]
        self.white_ball_init_pos = [[100, 200,270, 365], [100,200,385,485]]       #xmin xmax ymin ymax

        self.white_ball_color = [self.agent1_color, self.agent2_color]
        self.vis = 200
        self.vis_clear = 5

        self.max_n_hit = 3

        self.cross_color = 'green'
        self.total_reward = 0

        self.game_name = 'billiard'

    def reset(self):
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

        self.generate_map(self.map)
        self.merge_map()

        agent2idx = {}
        for idx, agent in enumerate(self.agent_list):
            if agent.type == 'agent':
                agent2idx[f'agent_{idx}'] = idx
            elif agent.type == 'ball':
                agent2idx[f'ball_{idx-2}'] = idx
            else:
                raise NotImplementedError
        self.agent2idx = agent2idx



        self.set_seed()
        self.init_state()
        self.step_cnt = 1
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False

        self.white_ball_in = [False, False]
        self.dead_agent_list = []
        self.total_reward = 0

        # self.hit_time_max = 10
        # self.now_hit = True
        # self.hit_time = 0
        # self.current_team = 0
        self.pre_num = len(self.agent_list)-1
        self.team_score = [0, 0]

        self.agent_energy = [self.agent_list[0].energy, self.agent_list[1].energy]
        # self.player1_n_hit = 1
        # self.player2_n_hit = 0

        self.white_ball_index = 0

        self.ball_left = [3, 3]
        # self.purple_ball_left = 3
        # self.green_ball_left = 3
        self.agent1_color = self.agent_list[0].color
        self.agent2_color = self.agent_list[1].color

        self.agent1_ball_color = self.agent1_color
        self.agent2_ball_color = self.agent2_color
        self.nonzero_reward_list = []
        self.output_reward = [0,0]
        self.score = [0,0]
        self.total_score = [0,0]

        self.num_ball_left = len(self.agent_list)-2
        self.pre_num_ball_left = len(self.agent_list)-2

        init_obs = self.get_obs()
        output_obs = self._build_from_raw_obs(init_obs)

        self.minimap_mode = False

        if self.minimap_mode:
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

            image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)

            output_obs[0]['minimap'] = image
            output_obs[1]['minimap'] = image

            # return [{"agent_obs": init_obs, "minimap":image}, {"agent_obs": np.zeros_like(init_obs)-1, "minimap":None}]

        return output_obs
        # return [init_obs, np.zeros_like(init_obs)-1]

    def check_overlap(self):
        pass

    def _idx2agent(self, idx):
        idx2agent = dict(zip(self.agent2idx.values(), self.agent2idx.keys()))
        return idx2agent[idx]

    def check_action(self, action_list):
        action = []

        for agent_idx in range(len(self.agent_list)):
            agent = self.agent_list[agent_idx]
            if agent.type == 'agent':
                if agent.color == self.agent1_color:
                    action.append(action_list[0])
                elif agent.color == self.agent2_color:
                    action.append(action_list[1])
                else:
                    raise NotImplementedError

                # _ = action_list.pop(0)
            else:
                action.append(None)

        return action

    def _check_ball_overlap(self, init_pos, init_r):
        for agent_idx, agent in enumerate(self.agent_list):
            pos = self.agent_pos[agent_idx]
            r = agent.r
            distance = (pos[0]-init_pos[0])**2 + (pos[1]-init_pos[1])**2
            if distance < (r + init_r)**2:
                return True
        return False


    def reset_cure_ball(self):      #fixme : random reset, need to check for overlap as well

        if self.white_ball_in[0] and self.white_ball_in[1]:
            new_agent_idx = [0,1]
        else:
            if self.white_ball_in[0]:
                new_agent_idx = [0]
            elif self.white_ball_in[1]:
                new_agent_idx = [1]
            else:
                raise NotImplementedError


        for idx in new_agent_idx:

            x_min, x_max, y_min, y_max = self.white_ball_init_pos[idx]

            random_init_pos_x = random.uniform(x_min, x_max)
            random_init_pos_y = random.uniform(y_min, y_max)

            #check for overlap
            while self._check_ball_overlap(init_pos=[random_init_pos_x, random_init_pos_y], init_r=15):
                random_init_pos_x = random.uniform(x_min, x_max)
                random_init_pos_y = random.uniform(y_min, y_max)


            new_agent = Agent(mass = 1, r = 15, position = [random_init_pos_x, random_init_pos_y],
                              color = self.white_ball_color[idx], vis = self.vis, vis_clear = self.vis_clear)
            self.agent_list.append(new_agent)
            self.white_ball_in[idx] = False

            new_boundary = self.get_obs_boundaray([random_init_pos_x, random_init_pos_y], 15, self.vis)
            self.obs_boundary_init.append(new_boundary)     #fixme: might has problem
            self.obs_boundary.append(new_boundary)
            self.agent_num += 1
            self.agent_pos.append([random_init_pos_x, random_init_pos_y])
            self.agent_v.append([0,0])
            self.agent_accel.append([0,0])
            init_obs = 0
            self.agent_theta.append([init_obs])
            self.agent_record.append([random_init_pos_x, random_init_pos_y])

            self.agent2idx[f'agent_{idx}'] = len(self.agent_list)-1




    def step(self, actions_list):

        input_action = self.check_action(actions_list)
        self.stepPhysics(input_action, self.step_cnt)
        self.cross_detect(self.agent_pos)
        self.output_reward = self._build_from_raw_reward()

        game_done = self.is_terminal()
        if not game_done:
            #reset white ball
            if np.logical_or(self.white_ball_in[0], self.white_ball_in[1]):
                self.reset_cure_ball()

        self.step_cnt += 1
        obs_next = self.get_obs()

        self.change_inner_state()
        self.record_energy()
        # pre_ball_left = self.ball_left
        self.clear_agent()
        # self.output_reward = self._build_from_raw_reward()


        # if not game_done:
        #     #reset white ball
        #     if np.logical_or(self.white_ball_in[0], self.white_ball_in[1]):
        #         self.reset_cure_ball()



        if self.minimap_mode:
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

        output_obs_next = self._build_from_raw_obs(obs_next)


        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return output_obs_next, self.output_reward, game_done, ''

    def _round_terminal(self):      #when all ball stop moving

        if self.hit_time <= self.hit_time_max:  #when player havent finish hitting
            if self.white_ball_in and self._all_ball_stop():
                return True, "WHITE BALL IN"        #white ball in when player is still at hitting time

            return False, "STILL AT HITTING TIME"
        else:
            if self.white_ball_in:
                if self._all_ball_stop():
                    return True, "WHITE BALL IN"
                else:
                    return False, "STILL MOVING"
            else:
                all_object_stop = self._all_object_stop()

                return all_object_stop, None

    def record_energy(self):
        for i,j in enumerate(self.agent_list):
            if j.type == 'agent':
                if j.color == self.agent1_color:
                    idx = 0
                elif j.color == self.agent2_color:
                    idx = 1

                self.agent_energy[idx] = j.energy



    def _all_object_stop(self):
        L = [(self.agent_v[agent_idx][0]**2 + self.agent_v[agent_idx][1]**2) < 1e-1 for agent_idx in range(self.agent_num)]
        return all(L)

    def _all_ball_stop(self):
        L = []
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'agent':
                continue
            L.append((self.agent_v[agent_idx][0]**2 + self.agent_v[agent_idx][1]**2) < 1e-1)
        return all(L)

    def cross_detect(self, new_pos):
        finals = []
        for object_idx in range(len(self.map['objects'])):
            object = self.map['objects'][object_idx]
            if object.can_pass() and object.color == self.cross_color:
                #arc_pos = object.init_pos
                finals.append(object)

        for agent_idx in range(len(self.agent_list)):
            agent = self.agent_list[agent_idx]
            agent_new_pos = new_pos[agent_idx]
            for final in finals:
                center = (final.init_pos[0] + 0.5*final.init_pos[2], final.init_pos[1]+0.5*final.init_pos[3])
                arc_r = final.init_pos[2]/2

                if final.check_radian(agent_new_pos, [0,0],0):
                    l = point2point(agent_new_pos, center)
                    if abs(l - arc_r) <= agent.r:
                        if agent.type == 'agent':
                            if agent.color == self.agent1_color:
                                self.white_ball_in[0] = True
                            elif agent.color == self.agent2_color:
                                self.white_ball_in[1] = True

                        # agent.color = self.cross_color
                        agent.finished = True
                        agent.alive = False
                        self.dead_agent_list.append(agent_idx)

    def clear_agent(self):
        if len(self.dead_agent_list) == 0:
            return
        index_add_on = 0

        self.score = [0,0]
        for idx in self.dead_agent_list:
            if self.agent_list[idx-index_add_on].name != 'agent':
                self.num_ball_left -= 1
            if self.agent_list[idx - index_add_on].type == 'ball':
                color = self.agent_list[idx - index_add_on].original_color
                if color == self.agent1_ball_color:
                    self.ball_left[0] -= 1
                    self.score[0] += 1
                    # self.purple_ball_left -= 1
                elif color == self.agent2_ball_color:
                    self.ball_left[1] -= 1
                    self.score[1] += 1
                    # self.green_ball_left -= 1
                self.agent2idx[self._idx2agent(idx-index_add_on)] = None
            #     pass
            # else:
            del self.agent_list[idx-index_add_on]
            del self.agent_pos[idx-index_add_on]
            del self.agent_v[idx-index_add_on]
            del self.agent_theta[idx-index_add_on]
            del self.agent_accel[idx-index_add_on]
            del self.obs_boundary_init[idx-index_add_on]
            del self.obs_boundary[idx-index_add_on]
            del self.obs_list[idx-index_add_on]

            # self.agent2idx[self._idx2agent(idx-index_add_on)] = None
            for name, id in self.agent2idx.items():
                if id is not None and id > (idx-index_add_on):
                    self.agent2idx[name] = id-1

            index_add_on += 1
        self.agent_num -= len(self.dead_agent_list)
        self.dead_agent_list = []


    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True


        if self.ball_left[0] <= 0 or self.ball_left[1] <= 0:
            return True

        return False


        # if self.step_cnt >= self.max_step:
        #     return True





        # if len(self.agent_list) == 1:       #all ball has been scored
        #     return True
        #
        #
        # if (self.player1_n_hit + self.player2_n_hit == self.max_n_hit*2):  #use up all the hit chance
        #     round_end, _ = self._round_terminal()
        #     if round_end:
        #         return True

        # return Fals

    def get_reward(self):
        # if len(self.agent_list) == 1 and not self.white_ball_in:
        #     return [500.]

        # reward = [0., 0.]
        reward = [int(i)*self.white_penalty for i in self.white_ball_in]
        for i in range(len(reward)):
            reward[i] += (self.score[i])*self.pot_reward

        self.score = [0,0]
        # if not self.white_ball_in:
        self.total_score[0] += reward[0]
        self.total_score[1] += reward[1]

        if self.is_terminal():
            winner = self.check_win()
            if winner == '0':
                reward[0] += 100
            elif winner == '1':
                reward[1] += 100


        reward[0] /= 100
        reward[1] /= 100

        return reward

    def check_win(self):
        total_reward = self.total_score
        if total_reward[0]>total_reward[1]:
            return '0'
        elif total_reward[0]<total_reward[1]:
            return '1'
        else:
            return '-1'


    def get_round_reward(self):
        #return only if the player's round is over
        round_end, _ = self._round_terminal()
        if round_end:
            round_reward = {"penalty": 0, "pot": (self.pre_num_ball_left-self.num_ball_left)*self.pot_reward}
            if self.white_ball_in:
                round_reward["penalty"] = self.white_penalty

            round_reward["total"] = round_reward['pot'] + round_reward["penalty"]
            return round_reward
        else:
            return None


    def _build_from_raw_reward(self):
        step_reward = self.get_reward()
        #step_reward = self.get_reward()
        # round_reawrd = self.get_round_reward()      #move it to if not done?
        return step_reward
        # _output_reward = {}
        # _output_reward[f"team_{self.current_team}"] = {"step_reward": step_reward, "round_reward": round_reawrd}
        # _output_reward[f"team_{1-self.current_team}"] = None
        # self.team_score[self.current_team] += round_reawrd['total'] if round_reawrd is not None else 0

        # return _output_reward

    def _build_from_raw_obs(self, next_obs):
        _output_obs_next = [0,0]

        for i,j in enumerate(self.agent_list):      #fixme: the transition of obs when cure ball is in need re-checking
            if j.type == 'agent':
                if j.color == self.agent1_color:
                    idx = 0
                elif j.color == self.agent2_color:
                    idx = 1

                try:
                    _output_obs_next[idx] = next_obs[i]
                except IndexError:
                    n = int(j.visibility/j.visibility_clear)
                    _output_obs_next[idx] = np.zeros((n,n)) -1
                    # _output_obs_next[0] = next_obs[i]
                # elif j.color == 'green':
                #     _output_obs_next[1] = next_obs[i]

        return [{"agent_obs":_output_obs_next[0], "id":"team_0"}, {"agent_obs": _output_obs_next[1], "id":"team_1"}]


    def render(self, info=None):

        if self.minimap_mode:
            pass
        else:

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
                obs_list, agent_list = self.align_obs()
                self.viewer.draw_view(obs_list, agent_list, leftmost_x=500, upmost_y=10, gap=100)
                # self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=10, gap = 100)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)


        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)

        # debug("No. of balls left: {}".format(self.agent_num-1), x = 100)
        debug(f"Agent1 ball left = {self.ball_left[0]}, total score = {self.total_score[0]}", c=self.agent1_color,x=  100, y =10)
        debug(f"Agent2 ball left = {self.ball_left[1]}, total score = {self.total_score[1]}", c=self.agent2_color,x = 100, y=30)

        # debug(f"-----------Current team {self.current_team}", x = 250)
        # debug('Current player:', x = 480, y = 145)
        # debug(f" Player 0 hit left = {self.max_n_hit-self.player1_n_hit}, Player 1 hit left = {self.max_n_hit-self.player2_n_hit}",
        #       x = 300, y = 100)

        # self.draw_table()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()

    def align_obs(self):
        obs_list = [0,0] #self.obs_list
        agent_list = [0,0] #self.agent_list

        for i,j in enumerate(self.agent_list):
            if j.type == 'agent':
                if j.color == self.agent1_color:
                    idx = 0
                elif j.color == self.agent2_color:
                    idx = 1

                obs_list[idx] = self.obs_list[i]
                agent_list[idx] = self.agent_list[i]

        return obs_list, agent_list


    def draw_table(self):

        debug("team 0", x = 130, y = 70, c=self.agent1_color)
        debug("No. breaks left: ", x = 20, y = 100 )
        debug(f"{self.max_n_hit-self.player1_n_hit}", x = 150, y = 100, c=self.agent1_color)
        debug('team 1', x = 190, y = 70, c= self.agent2_color)
        debug(f"{self.max_n_hit-self.player2_n_hit}", x = 210, y = 100, c=self.agent2_color)
        debug(f"Score: ", x=20, y=130)
        debug(f'{self.team_score[0]}', x = 150, y=130, c=self.agent1_color)
        debug(f'{self.team_score[1]}', x=210, y=130, c=self.agent2_color)

        pygame.draw.line(self.viewer.background, start_pos=[20, 90], end_pos=[230, 90], color=[0, 0, 0])
        pygame.draw.line(self.viewer.background, start_pos=[20, 120], end_pos=[230, 120], color=[0, 0, 0])
        pygame.draw.line(self.viewer.background, start_pos=[20, 150], end_pos=[230, 150], color=[0, 0, 0])

        pygame.draw.line(self.viewer.background, start_pos=[120, 60], end_pos=[120, 150], color=[0, 0, 0])
        pygame.draw.line(self.viewer.background, start_pos=[180, 60], end_pos=[180, 150], color=[0, 0, 0])

