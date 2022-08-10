from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import time
import pygame
import sys

class Running(OlympicsBase):
    def __init__(self, map, seed = None):
        self.minimap_mode = map['obs_cfg'].get('minimap', False)

        super(Running, self).__init__(map, seed)

        self.game_name = 'running'

        self.agent1_color = self.agent_list[0].color
        self.agent2_color = self.agent_list[1].color

        self.tau = map['env_cfg'].get('tau', 0.1)
        self.gamma = map["env_cfg"].get('gamma', 1)
        self.wall_restitution = map['env_cfg'].get('wall_restitution', 1)
        self.circle_restitution = map['env_cfg'].get('circle_restitution', 1)
        self.max_step = map['env_cfg'].get('max_step', 500)
        self.energy_recover_rate = map['env_cfg'].get('energy_recover_rate', 200)
        self.speed_cap = map['env_cfg'].get('speed_cap', 500)

        self.print_log = False
        self.print_log2 = False


        self.draw_obs = True
        self.show_traj = True

        #self.is_render = True

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
            # image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)

            # return [{"agent_obs": init_obs[0], "minimap":image}, {"agent_obs": init_obs[1], "minimap":image}]


        # return [{'agent_obs':init_obs[0]}, {'agent_obs':init_obs[1]}]

    def check_overlap(self):
        #todo
        pass

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]


        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 100.

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

        obs_next = self.get_obs()
        #self.check_overlap()
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

        # image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)

        # return image

    def check_win(self):
        if self.agent_list[0].finished and not (self.agent_list[1].finished):
            return '0'
        elif not(self.agent_list[0].finished) and self.agent_list[1].finished:
            return '1'
        else:
            return '-1'


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
                self.viewer.draw_obs(self.obs_boundary,         self.agent_list)

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



