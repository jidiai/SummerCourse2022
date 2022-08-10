from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer

import math


def closest_point(l1, l2, point):
    """
    compute the coordinate of point on the line l1l2 closest to the given point, reference: https://en.wikipedia.org/wiki/Cramer%27s_rule
    :param l1: start pos
    :param l2: end pos
    :param point:
    :return:
    """
    A1 = l2[1] - l1[1]
    B1 = l1[0] - l2[0]
    C1 = (l2[1] - l1[1])*l1[0] + (l1[0] - l2[0])*l1[1]
    C2 = -B1 * point[0] + A1 * point[1]
    det = A1*A1 + B1*B1
    if det == 0:
        cx, cy = point
    else:
        cx = (A1*C1 - B1*C2)/det
        cy = (A1*C2 + B1*C1)/det

    return [cx, cy]

def distance_to_line(l1, l2, pos):
    closest_p = closest_point(l1, l2, pos)

    n = [pos[0] - closest_p[0], pos[1] - closest_p[1]]  # compute normal
    nn = n[0] ** 2 + n[1] ** 2
    nn_sqrt = math.sqrt(nn)
    cl1 = [l1[0] - pos[0], l1[1] - pos[1]]
    cl1_n = (cl1[0] * n[0] + cl1[1] * n[1]) / nn_sqrt

    return abs(cl1_n)


class longjump(OlympicsBase):
    def __init__(self, map):
        super(longjump, self).__init__(map)

        self.jump = False

    def reset(self):
        self.init_state()
        self.step_cnt = 0
        self.done = False
        self.jump = False

        self.gamma = 0.98  # for longjump env

        self.viewer = Viewer()
        self._init_view()


    def cross_detect(self):
        """
        check whether the agent has reach the cross(final) line
        :return:
        """
        for agent_idx in range(self.agent_num):

            agent = self.agent_list[agent_idx]
            for object_idx in range(len(self.map['objects'])):
                object = self.map['objects'][object_idx]

                if not object.can_pass():
                    continue
                else:
                    #print('object = ', object.type)
                    if object.color == 'red' and object.check_cross(self.agent_pos[agent_idx], agent.r):

                        agent.color = 'red'
                        self.gamma = 0.85            #this will change the gamma for the whole env, so need to change if dealing with multi-agent
                        self.jump = True


    def step(self, actions_list):
        if self.jump:
            input_action = [None]       #if jump, stop actions
        else:
            input_action = actions_list

        self.stepPhysics(input_action)
        self.speed_limit()
        self.cross_detect()
        self.change_inner_state()
        self.step_cnt += 1

        step_reward = self.get_reward()
        obs_next = self.get_obs()
        done = self.is_terminal()

        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return obs_next, step_reward, done, ''

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].color == 'red' and (self.agent_v[agent_idx][0]**2 + self.agent_v[agent_idx][1]**2) < 1e-10:
                for object_idx in range(len(self.map['objects'])):
                    object = self.map['objects'][object_idx]
                    if object.color == 'red':
                        l1, l2 = object.init_pos
                        agent_reward[agent_idx] = distance_to_line(l1, l2, self.agent_pos[agent_idx])
        return agent_reward

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].color == 'red' and (
                    self.agent_v[agent_idx][0] ** 2 + self.agent_v[agent_idx][1] ** 2) < 1e-5:
                return True





