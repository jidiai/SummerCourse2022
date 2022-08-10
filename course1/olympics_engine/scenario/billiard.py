import numpy as np
from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
from olympics_engine.objects import Agent
import pygame
import sys
import math
import copy

def point2point(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class billiard(OlympicsBase):
    def __init__(self, map):
        self.minimap_mode = map['obs_cfg']['minimap']
        self.original_tau = map['env_cfg']['tau']
        self.faster = map['env_cfg']['faster']
        self.original_gamma = map['env_cfg']['gamma']

        super(billiard, self).__init__(map)

        self.tau = self.original_tau
        self.gamma = self.original_gamma
        self.wall_restitution = map['env_cfg']['wall_restitution']
        self.circle_restitution = map['env_cfg']['circle_restitution']
        self.max_step = map['env_cfg']['max_step']
        self.max_n_hit = map['env_cfg']['max_n_hit']
        self.white_penalty = map['env_cfg']['white_penalty']
        self.pot_reward = map['env_cfg']['pot_reward']

        self.print_log = False

        self.draw_obs = True
        self.show_traj = False

        self.dead_agent_list = []
        self.original_num_ball = len(self.agent_list)
        self.white_ball_in = False

        self.white_ball_init_pos = [100,375]
        self.white_ball_color = 'purple'
        self.vis = 200
        self.vis_clear = 5


        self.total_reward = 0

    def reset(self):
        self.tau = self.original_tau
        self.gamma = self.original_gamma

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

        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False

        self.white_ball_in = False
        self.dead_agent_list = []
        self.total_reward = 0

        self.hit_time_max = 10
        self.now_hit = True
        self.hit_time = 0
        self.current_team = 0
        self.pre_num = len(self.agent_list)-1
        self.team_score = [0, 0]
        init_obs = self.get_obs()

        self.player1_n_hit = 1
        self.player2_n_hit = 0

        self.white_ball_index = 0
        self.num_ball_left = 6
        self.pre_num_ball_left = 6

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

            return [{"agent_obs": init_obs, "minimap":image}, {"agent_obs": np.zeros_like(init_obs)-1, "minimap":None}]

        return [init_obs, np.zeros_like(init_obs)-1]

    def check_overlap(self):
        pass

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

        previous_pos = self.agent_pos

        actions_list = self.check_action(actions_list)
        if self.now_hit:
            self.tau = self.original_tau
            self.gamma = self.original_gamma

            input_action = actions_list
            self.hit_time += 1
            self.now_hit = (self.hit_time <= self.hit_time_max)
        else:
            input_action = [None for _ in range(len(self.agent_list))]
            self.tau = self.original_tau*self.faster
            self.gamma =  1-(1-self.original_gamma)*self.faster      #self.original_gamma**self.faster

        self.stepPhysics(input_action, self.step_cnt)

        self.cross_detect(self.agent_pos)

        game_done = self.is_terminal()



        self.step_cnt += 1

        output_reward = self._build_from_raw_reward()


        obs_next = self.get_obs()

        self.change_inner_state()

        self.clear_agent()
        #check overlapping
        #self.check_overlap()

        if not game_done:
            round_end, end_info = self._round_terminal()
            if not round_end:
                pass
            else:
                if end_info == "WHITE BALL IN":

                    new_agent = Agent(mass = 1, r = 15, position = self.white_ball_init_pos,
                                      color = self.white_ball_color, vis = self.vis,
                                      vis_clear = self.vis_clear)
                    self.white_ball_index = len(self.agent_list)
                    self.agent_list.append(new_agent)
                    new_boundary = self.get_obs_boundaray(self.white_ball_init_pos, 15, self.vis)
                    self.obs_boundary_init.append(new_boundary)
                    self.agent_num +=1
                    self.agent_pos.append(self.white_ball_init_pos)
                    self.agent_v.append([0,0])
                    self.agent_accel.append([0,0])
                    init_obs = 0
                    self.agent_theta.append([init_obs])
                    self.agent_record.append([self.white_ball_init_pos])
                    self.white_ball_in = False

                self.now_hit=True
                self.hit_time=0

                ball_in = (self.num_ball_left < self.pre_num_ball_left)
                if not ball_in or end_info=='WHITE BALL IN':     #if no pot or white ball penalty
                    self.current_team = 1 - self.current_team

                    if self.current_team == 0:
                        self.player1_n_hit += 1
                    elif self.current_team == 1:
                        self.player2_n_hit += 1
                    else:
                        raise NotImplementedError

                self.pre_num_ball_left = self.num_ball_left

                obs_next = self.get_obs()
                # round_score = len(self.agent_list)-self.pre_num
                # output_step_reward[self.current_team] += round_score *10        #reward for scoring





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
        return output_obs_next, output_reward, game_done, ''

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
            if object.can_pass() and object.color == 'blue':
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
                            self.white_ball_in=True
                        agent.color = 'blue'
                        agent.finished = True
                        agent.alive = False
                        self.dead_agent_list.append(agent_idx)

    def clear_agent(self):
        if len(self.dead_agent_list) == 0:
            return
        index_add_on = 0
        for idx in self.dead_agent_list:
            if self.agent_list[idx-index_add_on].name != 'agent':
                self.num_ball_left -= 1
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

            index_add_on += 1
        self.agent_num -= len(self.dead_agent_list)
        self.dead_agent_list = []


    def is_terminal(self):

        if len(self.agent_list) == 1:       #all ball has been scored
            return True


        if (self.player1_n_hit + self.player2_n_hit == self.max_n_hit*2):  #use up all the hit chance
            round_end, _ = self._round_terminal()
            if round_end:
                return True

        return False

    def get_reward(self):
        # if len(self.agent_list) == 1 and not self.white_ball_in:
        #     return [500.]

        reward = [0.]
        if not self.white_ball_in:
            reward[0] += len(self.dead_agent_list)*self.pot_reward
        else:
            for i in self.dead_agent_list:
                if self.agent_list[i].type == 'agent':      #penalise only once
                    reward[0] += self.white_penalty

        return reward



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
        round_reawrd = self.get_round_reward()      #move it to if not done?
        _output_reward = {}
        _output_reward[f"team_{self.current_team}"] = {"step_reward": step_reward, "round_reward": round_reawrd}
        _output_reward[f"team_{1-self.current_team}"] = None
        self.team_score[self.current_team] += round_reawrd['total'] if round_reawrd is not None else 0

        return _output_reward

    def _build_from_raw_obs(self, next_obs):
        _output_obs_next = [{},{}]
        next_obs = [x for x in next_obs if x is not None]
        if len(next_obs) == 0:
            next_obs = [np.zeros((40,40))-1]

        if self.minimap_mode:
            image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)

        _output_obs_next[self.current_team]["agent_obs"] = next_obs
        if self.minimap_mode:
            _output_obs_next[self.current_team]["minimap"] = image

        _output_obs_next[1-self.current_team]["agent_obs"] = [np.zeros_like(i)-1 for i in next_obs]
        if self.minimap_mode:
            _output_obs_next[1-self.current_team]["minimap"] = None

        # _output_obs_next[self.current_team] = next_obs
        # _output_obs_next[1-self.current_team] = [np.zeros_like(i)-1 for i in next_obs]
        return _output_obs_next


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
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=580, upmost_y=10)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)


        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)

        debug("No. of balls left: {}".format(self.agent_num-1), x = 100)
        # debug(f"-----------Current team {self.current_team}", x = 250)
        # debug('Current player:', x = 480, y = 145)
        # debug(f" Player 0 hit left = {self.max_n_hit-self.player1_n_hit}, Player 1 hit left = {self.max_n_hit-self.player2_n_hit}",
        #       x = 300, y = 100)

        self.draw_table()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()


    def draw_table(self):

        debug("team 0", x = 130, y = 70, c='purple')
        debug("No. breaks left: ", x = 20, y = 100 )
        debug(f"{self.max_n_hit-self.player1_n_hit}", x = 150, y = 100, c='purple')
        debug('team 1', x = 190, y = 70, c= 'green')
        debug(f"{self.max_n_hit-self.player2_n_hit}", x = 210, y = 100, c='green')
        debug(f"Score: ", x=20, y=130)
        debug(f'{self.team_score[0]}', x = 150, y=130, c='purple')
        debug(f'{self.team_score[1]}', x=210, y=130, c='green')

        pygame.draw.line(self.viewer.background, start_pos=[20, 90], end_pos=[230, 90], color=[0, 0, 0])
        pygame.draw.line(self.viewer.background, start_pos=[20, 120], end_pos=[230, 120], color=[0, 0, 0])
        pygame.draw.line(self.viewer.background, start_pos=[20, 150], end_pos=[230, 150], color=[0, 0, 0])

        pygame.draw.line(self.viewer.background, start_pos=[120, 60], end_pos=[120, 150], color=[0, 0, 0])
        pygame.draw.line(self.viewer.background, start_pos=[180, 60], end_pos=[180, 150], color=[0, 0, 0])

