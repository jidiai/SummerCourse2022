import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)
from collections import deque, namedtuple
import random
import math
import copy
# from olympics_engine.generator import create_scenario
# from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition


# map_id = random.randint(1,4)
# map_id = 1
# Gamemap = create_scenario('running-competition')
# env = Running_competition(meta_map=Gamemap,map_id=map_id, vis = 200, vis_clear=5, agent1_color = 'light red',
#                           agent2_color = 'blue')

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import APPOTrainer
import gym
from gym.spaces.box import Box
from gym.spaces.multi_discrete import MultiDiscrete

COLORS = {
    'red': [255,0,0],
    'light red': [255, 127, 127],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'orange': [255, 127, 0],
    'grey':  [176,196,222],
    'purple': [160, 32, 240],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'light green': [204, 255, 229],
    'sky blue': [0,191,255],
    # 'red-2': [215,80,83],
    # 'blue-2': [73,141,247]
}

COLOR_TO_IDX = {
    'light green': 0,
    'green': 1,
    'sky blue': 2,
    'orange': 3,
    'grey': 4,
    'purple': 5,
    'black': 6,
    'red': 7,
    'blue':8,
    'white': 9,
    'light red': 10
    # 'red-2': 9,
    # 'blue-2': 10
}

IDX_TO_COLOR = {
    0: 'light green',
    1: 'green',
    2: 'sky blue',
    3: 'orange',
    4: 'grey',
    5: 'purple',
    6: 'black',
    7: 'red',
    8: 'blue',
    9: 'white',
    10: 'light red'
    # 9: 'red-2',
    # 10: 'blue-2'
}

# Map of object type to integers
OBJECT_TO_IDX = {
    'agent': 0,
    'wall': 1,  # 反弹
    'cross': 2,   # 可穿越
    'goal': 3,   # 可穿越 # maybe case by case
    'arc': 4,
    'ball': 5
}

import math


# get obs func
def dot(vec_1, vec_2):
    """
    计算点乘，vec_1, vec_2都为向量
    """
    return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]


def cross(vec_1, vec_2):
    """
    计算叉积，vec_1, vec_2都为向量
    """
    return vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]


def distance_2points(vec):
    """
    计算两个点直接距离，vec为向量
    """
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def rotate(x, y, theta):
    """
    坐标轴转，物体不转
    formula reference: https://www.cnblogs.com/jiahenhe2/p/10135235.html
    """
    x_n = math.cos(theta * math.pi / 180) * x + math.sin(theta * math.pi / 180) * y
    y_n = -math.sin(theta * math.pi / 180) * x + math.cos(theta * math.pi / 180) * y
    return x_n, y_n


def rotate2(x, y, theta):
    """
    坐标轴不转，物体转; 坐标点旋转后的点坐标, 逆时针旋转theta
    """
    x_n = math.cos(theta * math.pi / 180) * x + math.sin(theta * math.pi / 180) * y
    y_n = -math.sin(theta * math.pi / 180) * x + math.cos(theta * math.pi / 180) * y
    return x_n, y_n



def get_distance(AB, vec_OC, AB_length, pixel):
    """
    通过向量叉乘，求点C到线段AB的距离; 通过点乘判断点位置的三种情况，左边、右边和上面
    :param: 两个点AB -> [[].[]]，点C->[]，AB线段长度
    :return:
    formula reference: https://blog.csdn.net/qq_45735851/article/details/114448767
    """
    vec_OA, vec_OB = AB[0], AB[1]
    vec_CA = [vec_OA[0] - vec_OC[0], vec_OA[1] - vec_OC[1]]
    vec_CB = [vec_OB[0] - vec_OC[0], vec_OB[1] - vec_OC[1]]
    vec_AB = [vec_OB[0] - vec_OA[0], vec_OB[1] - vec_OA[1]]

    vec_AC = [-vec_OA[0] + vec_OC[0], -vec_OA[1] + vec_OC[1]]
    vec_BC = [-vec_OB[0] + vec_OC[0], -vec_OB[1] + vec_OC[1]]

    if pixel:
        if dot(vec_AB, vec_AC) < 0:
            d = distance_2points(vec_AC)
        elif dot(vec_AB, vec_BC) > 0:
            d = distance_2points(vec_BC)
        else:
            d = math.ceil(cross(vec_CA, vec_CB) / AB_length)
    else:
        d = math.ceil(cross(vec_CA, vec_CB) / AB_length)
    return d


def get_obs_check_radian(start_radian, end_radian, angle):
    if start_radian >= 0:
        if end_radian >= 0 and end_radian >= start_radian:
            return True if (start_radian <= angle <= end_radian) else False
        elif end_radian >= 0 and end_radian < start_radian:
            return True if not (start_radian <= angle <= end_radian) else False

        elif end_radian <= 0:
            if angle >= 0 and angle >= start_radian:
                return True
            elif angle < 0 and angle <= end_radian:
                return True
            else:
                return False

    elif start_radian < 0:
        if end_radian >= 0:
            if angle >= 0 and angle < end_radian:
                return True
            elif angle < 0 and angle > start_radian:
                return True
            else:
                return False
        elif end_radian < 0 and end_radian > start_radian:
            return True if (angle < 0 and start_radian <= angle <= end_radian) else False
        elif end_radian < 0 and end_radian < start_radian:
            return True if not (end_radian <= angle <= start_radian) else False




# others


def point2line(l1, l2, point):
    """
    :param l1: coord of line start point
    :param l2: coord of line end point
    :param point: coord of circle center
    :return:
    """

    l1l2 = [l2[0] - l1[0], l2[1]-l1[1]]
    l1c = [point[0]-l1[0], point[1]-l1[1]]

    cross_prod = abs(l1c[0]*l1l2[1] - l1c[1]*l1l2[0])

    l1l2_length = math.sqrt(l1l2[0]**2 + l1l2[1]**2)
    return cross_prod/l1l2_length


def cross_prod(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]


def line_intersect(line1, line2, return_p = False):       #[[x1,y1], [x2,y2]], https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    p = line1[0]
    r = [line1[1][0] - line1[0][0] , line1[1][1] - line1[0][1]]

    q = line2[0]
    s = [line2[1][0] - line2[0][0] , line2[1][1] - line2[0][1]]

    rs = cross_prod(r,s)
    if rs == 0:
        return False
    else:
        q_p = [q[0]-p[0], q[1]-p[1]]

        t = cross_prod(q_p, s)/rs
        u = cross_prod(q_p, r)/rs

        if 0<=t<=1 and 0<=u<=1:
            point = [p[0]+t*r[0], p[1]+t*r[1]]
            if return_p:
                return point
            else:
                return True
        else:
            return False

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

    distance = (closest_p[0]-pos[0])**2 + (closest_p[1]-pos[1])**2
    distance = math.sqrt(distance)
    return distance
    # return math.sqrt(distance)
    #
    # n = [pos[0] - closest_p[0], pos[1] - closest_p[1]]  # compute normal
    # nn = n[0] ** 2 + n[1] ** 2
    # nn_sqrt = math.sqrt(nn)
    # cl1 = [l1[0] - pos[0], l1[1] - pos[1]]
    # cl1_n = (cl1[0] * n[0] + cl1[1] * n[1]) / nn_sqrt
    #
    # assert distance == abs(cl1_n), print(f'distance = {distance}, cl1n = {abs(cl1_n)}')
    #
    # return abs(cl1_n)



#### new get_obs
def point_rotate(center, point, theta):
    px = point[0] - center[0]
    py = point[1] - center[1]

    nx = [math.cos(theta * math.pi / 180), math.sin(theta * math.pi / 180)]
    ny = [-math.sin(theta * math.pi / 180), math.cos(theta * math.pi / 180)]
    new_x = px * nx[0] + py * nx[1]
    new_y = px * ny[0] + py * ny[1]
    return [new_x, new_y]

def DDA_line(matrix, draw_line, vis, vis_clear, value, view_back):
    size = int(vis/vis_clear)
    assert matrix.shape[0] == size
    if len(draw_line) == 1:
        point1 = draw_line[0]
        x1, y1 = point1
        x1 += view_back
        y1 += vis/2
        x1 /= vis_clear
        y1 /= vis_clear

        x = x1-0.5
        y = y1-0.5
        matrix[size-1-int(x)][int(y)] = 1
        return matrix

    elif len(draw_line) == 2:
        point1, point2 = draw_line
    else:
        raise NotImplementedError

    x1,y1 = point1
    x1 += view_back
    y1 += vis/2
    x1 /= vis_clear
    y1 /= vis_clear
    x2, y2 = point2
    x2 += view_back
    y2 += vis/2
    x2 /= vis_clear
    y2 /= vis_clear

    dx = x2-x1
    dy = y2-y1

    if abs(dx) > abs(dy):
        steps = abs(dx)
    else:
        steps = abs(dy)

    if steps == 0:  #numerical error at avoiding intersection point repetition
        point1 = draw_line[0]
        x1, y1 = point1
        x1 += view_back
        y1 += vis/2
        x1 /= vis_clear
        y1 /= vis_clear

        x = x1-0.5
        y = y1-0.5
        matrix[size-1-int(x)][int(y)] = 1
        return matrix

        # raise NotImplementedError


    delta_x = float(dx/steps)
    delta_y = float(dy/steps)

    x = x1-0.5
    y = y1-0.5

    assert  0<=int(x)<=size-1
    assert 0<=int(y)<=size-1

    for i in range(0, int(steps + 1)):
        if not 0<=size-1-int(x)<size or not 0<=int(y) <size:
            raise NotImplementedError
        matrix[size-1-int(x)][int(y)] = value


        x += delta_x
        y += delta_y

    return matrix




# gamemap = {'objects':[], 'agents':[]}

# gamemap['objects'].append(Wall(init_pos=[[50, 200], [90, 600]], length = None, color = 'black'))



class EnvWrapped(gym.Env):

    def __init__(self, config):

        self.action_space = Box(low=np.array([-100, -30]), high=np.array([200,30]), dtype=np.float32)
        self.observation_space = MultiDiscrete(np.ones(1600)*11)  #MultiDiscrete(np.ones(40,40)*10)
        self.cur_obs = None
        self.episode_len = 0
        self.ctrl_agent_index = 0
        self._build_Gamemap()

        self.env.max_episode_steps = 500
        # Gamemap = create_scenario('running-competition')
        # self.env = Running_competition(meta_map=Gamemap,map_id=map_id, vis = 200, vis_clear=5, agent1_color = 'light red',
        #                   agent2_color = 'blue')

    def _build_Gamemap(self):
        gamemap = {'objects':[], 'agents':[]}
        Wall = EnvWrapped.Wall
        Arc = EnvWrapped.Arc
        Cross = EnvWrapped.Cross
        Agent = EnvWrapped.Agent
        gamemap['objects'].append(Wall(init_pos=[[50, 150], [50, 300]], length = None, color = 'black'))
        gamemap['objects'].append(Wall(init_pos=[[50, 300], [250, 300]], length = None, color = 'black'))
        gamemap['objects'].append(Wall(init_pos=[[250, 300], [250, 650]], length = None, color = 'black'))
        gamemap['objects'].append(Wall(init_pos=[[250, 650], [450, 650]], length = None, color = 'black'))
        gamemap['objects'].append(Wall(init_pos=[[450, 300], [450, 650]], length = None, color = 'black'))
        gamemap['objects'].append(Wall(init_pos=[[450, 300], [650, 300]], length = None, color = 'black'))
        gamemap['objects'].append(Wall(init_pos=[[50, 150], [650, 150]], length = None, color = 'black'))
        gamemap['objects'].append(Cross(init_pos=[[650,300], [650, 150]], color = 'red'))
        gamemap['objects'].append(Cross(init_pos=[[325,500], [350, 475]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[375,500], [350, 475]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[325,400], [350, 375]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[375,400], [350, 375]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[325,330], [350, 305]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[375, 330], [350, 305]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[325,600], [350, 575]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[375,600], [350, 575]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[100,200], [125, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[100,250], [125, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[200,200], [225, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[200,250], [225, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[300,250], [325, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[400,200], [425, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[400,250], [425, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[500,200], [525, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[500,250], [525, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[600,200], [625, 225]], color = 'grey'))
        gamemap['objects'].append(Cross(init_pos=[[600,250], [625, 225]], color = 'grey'))
        gamemap['agents'].append(Agent(position=[400, 60], r = 20, mass = 1, vis=200, vis_clear=5))
        gamemap['agents'].append(Agent(position=[300, 600], r = 20, mass = 1, vis=200, vis_clear=5))
        gamemap['view'] = {'width': 600, 'height':600, 'edge': 50, 'init_obs': [-90,-90]}

        map_id = random.randint(1,4)
        map_id = 1
        self.env = EnvWrapped.Running_competition(gamemap, map_id=3, vis=200, vis_clear=5, agent1_color = 'light red', agent2_color='blue')



    @staticmethod
    def Wall(init_pos, length=None, color='black', ball_can_pass=False, width=None):
        class GameObj(object):
            """
            Base class for olympic engine objects
            """
            def __init__(self, type, color):
                assert type in OBJECT_TO_IDX, type
                assert color in COLOR_TO_IDX, color
                self.type = type
                self.color = color
                self.name = None
                self.contains = None
                # Initial position of the object
                self.init_pos = None

                # Current position of the object
                self.cur_pos = None

            def can_pass(self):
                """ 是否能穿越"""
                return False

            def can_bounce(self):
                """ 是否能反弹 """
                return False

            def render(self):
                """Draw this object with the given renderer"""
                raise NotImplementedError
        class Wall(GameObj):
            def __init__(self, init_pos, length = None, color="black",ball_can_pass = False, width = None):
                super(Wall, self).__init__(type='wall', color=color)
                self.init_pos = init_pos
                if length is None or length == "None":
                    l1,l2 = self.init_pos
                    self.length = math.sqrt((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)
                else:
                    self.length = length
                self.width = 2 if width is None else width
                self.ball_can_pass = ball_can_pass

                self.fn()
                self.wall = 'wall'
                self.l1, self.l2 = self.init_pos

            def fn(self):
                """
                Ax + By = C
                """
                l1, l2 = self.init_pos
                self.A = l1[1] - l2[1]
                self.B = l2[0] - l1[0]
                self.C = (l1[1] - l2[1])*l1[0] + (l2[0] - l1[0]) * l1[1]

            def check_on_line(self, point):
                temp = self.A * point[0] + self.B * point[1]
                #if temp == self.C:     #on the line or at least the line extension
                if abs(temp-self.C) <= 1e-6:
                    if not ((min(self.l1[0], self.l2[0]) <= point[0] <= max(self.l1[0], self.l2[0])) and (
                            min(self.l1[1], self.l2[1]) <= point[1] <= max(self.l1[1], self.l2[1]))):
                        return False        #not on the line segment
                    else:
                        return True
                return False

            def collision_time(self, pos, v, radius, add_info = None):
                """
                compute the collision time (line-circle collision and endpoint-circle collision)
                :param pos:  position of the circle
                :param v:  velocity of the circle
                :param radius:
                :return:  wall_col_t, col_target('wall' or 'l1')
                """

                closest_p = closest_point(l1=self.l1, l2=self.l2, point = pos)
                n = [pos[0] - closest_p[0], pos[1] - closest_p[1]]  # compute normal
                nn = n[0] ** 2 + n[1] ** 2
                nn_sqrt = math.sqrt(nn)
                cl1 = [self.l1[0] - pos[0], self.l1[1] - pos[1]]
                cl1_n = cl1[0] * n[0] + cl1[1] * n[1]
                v_n = (n[0] * v[0] + n[1] * v[1])

                if v_n == 0:
                    return -1, None

                r_ = radius if cl1_n < 0 else -radius
                wall_col_t = cl1_n/v_n + r_ * nn_sqrt/v_n       #the collision time with the line segment

                #check the collision point is on the line segment
                new_pos = [pos[0] + wall_col_t * v[0], pos[1] + wall_col_t * v[1] ]
                collision_point = closest_point(self.l1, self.l2, new_pos)
                on_the_line = self.check_on_line(collision_point)

                if on_the_line:
                    return wall_col_t, 'wall'       #if the collision point is on the line segment, the circle will collide with the line, no need to check the end point collision
                else:
                    wall_col_t = -1

                #now check the endpoint collision
                tl1 = self._endpoint_collision_time(pos, v, radius, self.l1)
                tl2 = self._endpoint_collision_time(pos, v, radius, self.l2)

                if tl1 >= 0 and tl2 >= 0:
                    t_endpoint = min(tl1, tl2)
                    endpoint_target = 'l1' if tl1 < tl2 else 'l2'
                elif tl1<0 and tl2<0:
                    t_endpoint = -1
                    endpoint_target = None
                else:
                    t_endpoint = tl1 if tl1>=0 else tl2
                    endpoint_target = 'l1' if tl1 >=0 else 'l2'

                #no need to compare with the wall col t
                return t_endpoint, endpoint_target


            def _endpoint_collision_time(self, pos, v, radius, endpoint):

                deno = v[0]**2 + v[1]**2
                k = ((pos[0]-endpoint[0])*v[0] + (pos[1]-endpoint[1])*v[1])/deno
                c = (radius**2 - (pos[0]-endpoint[0])**2 - (pos[1]-endpoint[1])**2)/deno
                sqrt = c+k**2
                if sqrt < 0:
                    tl = -1        # will not collide with this endpoint
                else:
                    sqrt = math.sqrt(sqrt)
                    t1 = -k+sqrt
                    t2 = -k-sqrt

                    if t1>=0 and t2>=0:
                        tl = min(t1, t2)
                    elif t1<0 and t2<0:
                        tl = -1
                    elif t1 >=0 and t2 <0:
                        tl = t1
                    else:
                        raise NotImplementedError("endpoint collision time error")

                return tl

            def collision_response(self, pos, v, radius, col_target, col_t, restitution = 1):
                """
                compute collision response with the wall or the endpoint
                :param pos:
                :param v:
                :param radius:
                :param col_target: collision target
                :param col_t: collision time
                :param restitution:
                :return:
                """

                if col_target == 'wall':
                    closest_p = closest_point(l1=self.l1, l2=self.l2, point=pos)
                    n = [pos[0] - closest_p[0], pos[1] - closest_p[1]]  # compute normal
                    nn = n[0] ** 2 + n[1] ** 2
                    v_n = (n[0] * v[0] + n[1] * v[1])

                    factor = 2 * v_n / nn
                    vx_new = v[0] - factor * n[0]
                    vy_new = v[1] - factor * n[1]

                elif col_target == 'l1' or 'l2':
                    l = self.l1 if col_target == 'l1' else self.l2

                    n = [pos[0] - l[0], pos[1] - l[1]]
                    v_n = v[0]*n[0] + v[1]*n[1]
                    nn = n[0]**2 + n[1]**2
                    factor = 2 * v_n/nn
                    vx_new = v[0] - factor * n[0]
                    vy_new = v[1] - factor * n[1]

                else:
                    raise NotImplementedError("collision response error")

                col_x = pos[0] + v[0] * col_t
                col_y = pos[1] + v[1] * col_t

                return [col_x, col_y], [vx_new*restitution, vy_new*restitution]


            def can_bounce(self):
                return True

            def render(self):
                # todo
                pass
        return Wall(init_pos, length, color, ball_can_pass, width)

    @staticmethod
    def Arc(init_pos, start_radian, end_radian, color, passable, collision_mode, width=None):
        class GameObj(object):
            """
            Base class for olympic engine objects
            """
            def __init__(self, type, color):
                assert type in OBJECT_TO_IDX, type
                assert color in COLOR_TO_IDX, color
                self.type = type
                self.color = color
                self.name = None
                self.contains = None
                # Initial position of the object
                self.init_pos = None

                # Current position of the object
                self.cur_pos = None

            def can_pass(self):
                """ 是否能穿越"""
                return False

            def can_bounce(self):
                """ 是否能反弹 """
                return False

            def render(self):
                """Draw this object with the given renderer"""
                raise NotImplementedError
        class Arc(GameObj):
            def __init__(self, init_pos, start_radian, end_radian,color, passable, collision_mode, width=None):     #collision_mode:  0----collide at start point
                super(Arc, self).__init__(type =  'arc', color = color)                                 #                 1----collide at end point
                self.init_pos = init_pos        #[x,y,width, height]                                    #                 2----collide at both point
                #                 3----no endpoint collision
                start_radian = start_radian*math.pi/180
                end_radian = end_radian * math.pi/180

                if start_radian < -math.pi or start_radian > math.pi or end_radian < -math.pi or end_radian > math.pi:
                    raise ValueError("The arc radian should be within the range [-pi, pi]")

                #if start_radian > end_radian:
                #    raise ValueError("The starting radian should be less than the end radian")
                self.start_radian, self.end_radian = start_radian, end_radian

                self.width = 2 if width is None else width

                self.passable = passable

                self.center = [init_pos[0]+1/2*init_pos[2], init_pos[1]+1/2*init_pos[3]]
                if init_pos[2] == init_pos[3]:
                    self.circle = True
                    self.R = 1/2*init_pos[2]
                else:
                    self.circle = False     #ellipse

                self.ball_can_pass = False
                self.arc = 'arc'
                self.collision_mode = collision_mode
                assert self.collision_mode in [0,1,2,3], print('ERROR: collision_mode of arc is wrong!')


            def collision_response(self, pos, v, r, col_target, t, restitution = 1):

                x_old, y_old = pos
                x_new = x_old + v[0]*t
                y_new = y_old + v[1]*t
                n = [x_new - self.center[0], y_new - self.center[1]]

                v_n = v[0] * n[0] + v[1] * n[1]
                nn = n[0] ** 2 + n[1] ** 2
                factor = 2 * v_n / nn
                vx_new = v[0] - factor * n[0]
                vy_new = v[1] - factor * n[1]

                col_x = pos[0] + v[0]*t
                col_y = pos[1] + v[1]*t

                return [col_x, col_y], [vx_new*restitution, vy_new*restitution]

            def collision_time(self, pos, v, radius, add_info):
                #print('pos = {}, v = {}, center = {}, R = {}, r = {}'.format(pos, v, self.center, self.R, radius))
                #if circle
                if self.circle:

                    #else, compute the collision time and the target
                    cx, cy = self.center
                    x,y = pos
                    vx, vy = v
                    l = vx*x - cx*vx + y*vy - cy*vy
                    k = vx**2 + vy**2
                    h = (x**2 + y**2) + (cx**2 + cy**2) - 2*(cx*x + y*cy)


                    #time of colliding with inner circle
                    RHS = (l/k)**2 - h/k + ((self.R - radius)**2)/k

                    if RHS < 0:
                        #print('inner collision has no solution')
                        t1 = -1
                    else:
                        sqrt =  math.sqrt(RHS)
                        t_inner1 = -(l/k) + sqrt
                        t_inner2 = -(l/k) - sqrt

                        if abs(t_inner1) <= 1e-10:
                            t_inner1 = 0
                        if abs(t_inner2) <= 1e-10:
                            t_inner2 = 0

                        t1_check = self.check_radian(pos, v, t_inner1)

                        t2_check = self.check_radian(pos, v, t_inner2)

                        if t1_check and t2_check:

                            if abs(t_inner1) < 1e-10 and [add_info[0], add_info[1], 0.] in add_info[2]:
                                t1 = t_inner2
                            elif abs(t_inner2) <1e-10 and [add_info[0], add_info[1], 0.] in add_info[2]:
                                t1 = t_inner1
                            else:

                                if t_inner1 <= 0 and t_inner2 <= 0:
                                    t1 = max(t_inner1, t_inner2)
                                elif t_inner1 >= 0 and t_inner2 >= 0:
                                    t1 = min(t_inner1, t_inner2)
                                elif t_inner1 >= 0 and t_inner2 <= 0:
                                    t1 = t_inner1
                                else:
                                    #print('CHECK t1 = {}, t2 = {}'.format(t_inner1, t_inner2))
                                    raise NotImplementedError

                        elif t1_check and not t2_check:
                            t1 = t_inner1

                        elif not t1_check and t2_check:
                            t1 = t_inner2

                        else:       #when both collision is outside the arc angle range
                            t1 = -1

                        #print('Inner time = {} ({}) and {}({}); t1 = {}'.format(t_inner1, t1_check, t_inner2, t2_check, t1))


                    #time of colliding with outter circle
                    RHS2 = (l/k)**2 - h/k + (self.R + radius)**2/k

                    if RHS2 < 0:
                        #print('outter collision has no solution')
                        t2 = -1
                    else:
                        sqrt2  = math.sqrt(RHS2)
                        t_outter1 = -(l/k) + sqrt2
                        t_outter2 = -(l/k) - sqrt2

                        if abs(t_outter1) <= 1e-10:
                            t_outter1 = 0
                        if abs(t_outter2) <= 1e-10:
                            t_outter2 = 0

                        #check radian, for both t,
                        t1_check = self.check_radian(pos, v, t_outter1)
                        t2_check = self.check_radian(pos, v, t_outter2)

                        if t1_check and t2_check:

                            if abs(t_outter1) < 1e-10 and [add_info[0], add_info[1], 0.] in add_info[2]:
                                t2 = t_outter2
                            elif abs(t_outter2) <1e-10 and [add_info[0], add_info[1], 0.] in add_info[2]:
                                t2 = t_outter1
                            else:
                                if t_outter1 <= 0 and t_outter2 <= 0:     #leaving the collision point
                                    t2 = max(t_outter1, t_outter2)
                                elif t_outter1 >= 0 and t_outter2 >= 0:       #approaching the collision point
                                    t2 = min(t_outter1, t_outter2)
                                elif t_outter1 >= 0 and t_outter2 <= 0:       #inside the circle
                                    t2 = t_outter1
                                else:
                                    raise NotImplementedError

                        elif t1_check and not t2_check:
                            t2 = t_outter1

                        elif not t1_check and t2_check:
                            t2 = t_outter2

                        else:
                            t2 = -1

                        #print('Outter time = {}({}) and {}({}); t2 = {}'.format(t_outter1,t1_check, t_outter2, t2_check,t2))

                    if t1 >= 0 and t2 >= 0:

                        if t1 > t2:
                            col_target = 'outter'
                            col_t = t2
                        elif t1 < t2:
                            col_target = 'inner'
                            col_t = t1
                        else:
                            #print('t1 = {}, t2 = {}'.format(t1, t2))
                            raise NotImplementedError
                    elif t1 < 0 and t2 >= 0:
                        col_target = 'outter'
                        col_t = t2
                    elif t1 >= 0 and t2 < 0:
                        col_target = 'inner'
                        col_t = t1

                    else:
                        col_target = None
                        col_t = -1

                    #print('Collision time  = {}, target = {}'.format(col_t, col_target))


                    return col_t, None if col_target is None else 'arc'

                else:       #ellipse arc collision

                    raise NotImplementedError("The collision of ellipse wall is not implemented yet...")



            def check_radian(self, pos, v, t):      #this is to check the collision is within degree range of the arc

                x_old, y_old = pos
                x_new = x_old + v[0]*t
                y_new = y_old + v[1]*t      #compute the exact collision position

                angle = math.atan2(self.center[1] - y_new, x_new - self.center[0])      #compute the angle of the circle, which is also the angle of the collision point

                #print('current angle ={}, start = {} , end = {}; pos = {}'.format(angle, self.start_radian, self.end_radian, [x_new-self.center[0], y_new-self.center[1]]))

                #collision at endpoint
                if self.collision_mode == 0 and angle==self.start_radian:        #collide at start point
                    return True
                if self.collision_mode == 1 and angle==self.end_radian:         #collide at end point
                    return True
                if self.collision_mode == 2 and (angle==self.start_radian or angle==self.end_radian):       #collide at both points
                    return True


                if self.start_radian >= 0:
                    if self.end_radian >= 0 and self.end_radian >= self.start_radian:
                        return True if (self.start_radian < angle < self.end_radian) else False

                    elif self.end_radian >= 0 and self.end_radian < self.start_radian:

                        return True if not(self.start_radian < angle < self.end_radian) else False


                    elif self.end_radian <= 0:

                        if angle >= 0 and angle > self.start_radian:
                            return True
                        elif angle < 0 and angle < self.end_radian:
                            return True
                        else:
                            return False


                elif self.start_radian < 0:

                    if self.end_radian >= 0:

                        if angle >= 0 and angle < self.end_radian:
                            return True
                        elif angle < 0 and angle > self.start_radian:
                            return True
                        else:
                            return False

                    elif self.end_radian < 0 and self.end_radian > self.start_radian:

                        return True if (angle < 0 and self.start_radian < angle < self.end_radian) else False

                    elif self.end_radian < 0 and self.end_radian < self.start_radian:

                        return True if not( self.end_radian < angle < self.start_radian ) else False

                else:
                    pass

            def check_inside_outside(self, pos, v, t):      #this is to determine the circle is collision from inside or outside

                x_old, y_old = pos
                x_new = x_old + v[0]*t
                y_new = y_old + v[1]*t

                #todo
                #check whether the line from pos_old to pos_new intersect with the arc, if they do, then this t penertrate the arc

                pass

            def check_on_line(self):
                pass

            def can_pass(self):
                if self.passable:
                    return True
                else:
                    return False

            def can_bounce(self):
                if self.passable:
                    return False
                else:
                    return True
        return Arc(init_pos, start_radian, end_radian, color, passable, collision_mode, width)

    @staticmethod
    def Cross(init_pos, length=None, color='red', width=None, ball_can_pass=True):
        class GameObj(object):
            """
            Base class for olympic engine objects
            """
            def __init__(self, type, color):
                assert type in OBJECT_TO_IDX, type
                assert color in COLOR_TO_IDX, color
                self.type = type
                self.color = color
                self.name = None
                self.contains = None
                # Initial position of the object
                self.init_pos = None

                # Current position of the object
                self.cur_pos = None

            def can_pass(self):
                """ 是否能穿越"""
                return False

            def can_bounce(self):
                """ 是否能反弹 """
                return False

            def render(self):
                """Draw this object with the given renderer"""
                raise NotImplementedError
        class Cross(GameObj):
            def __init__(self, init_pos, length=None, color="red", width = None, ball_can_pass = True):
                super(Cross, self).__init__(type='cross', color=color)
                self.init_pos = init_pos
                if length is None:
                    l1,l2 = self.init_pos
                    self.length = math.sqrt((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)
                else:
                    self.length = length

                if color == 'red':
                    self.width = 5
                else:
                    self.width = 2 if width is None else width

                self.l1, self.l2 = self.init_pos
                self.fn()

            def fn(self):
                """
                Ax + By = C
                """
                l1, l2 = self.init_pos
                self.A = l1[1] - l2[1]
                self.B = l2[0] - l1[0]
                self.C = (l1[1] - l2[1]) * l1[0] + (l2[0] - l1[0]) * l1[1]

            def check_on_line(self, point):
                temp = self.A * point[0] + self.B * point[1]
                #if temp == self.C:     #on the line or at least the line extension
                if abs(temp-self.C) <= 1e-6:
                    if not ((min(self.l1[0], self.l2[0]) <= point[0] <= max(self.l1[0], self.l2[0])) and (
                            min(self.l1[1], self.l2[1]) <= point[1] <= max(self.l1[1], self.l2[1]))):
                        return False        #not on the line segment
                    else:
                        return True
                return False

            def check_on_line2(self, point):  # or on the extension of the line segment
                temp = self.A * point[0] + self.B * point[1]
                return True if temp == self.C else False


            def check_cross(self, pos, radius, return_dist = False):
                l1, l2 = self.init_pos
                closest_p = closest_point(l1= l1, l2=l2, point = pos)
                if not ((min(l1[0], l2[0]) <= closest_p[0] <= max(l1[0], l2[0])) and (
                        min(l1[1], l2[1]) <= closest_p[1] <= max(l1[1], l2[1]))):
                    #print('THE CROSSING POINT IS NOT ON THE CROSS LINE SEGMENT')
                    return False

                n = [pos[0] - closest_p[0], pos[1] - closest_p[1]]  # compute normal
                nn = n[0] ** 2 + n[1] ** 2
                nn_sqrt = math.sqrt(nn)
                cl1 = [l1[0] - pos[0], l1[1] - pos[1]]
                cl1_n = (cl1[0] * n[0] + cl1[1] * n[1])/nn_sqrt

                if return_dist:
                    return abs(cl1_n) - radius

                if abs(cl1_n) - radius <= 0:
                    return True
                else:
                    return False


            def can_pass(self):
                """ 是否能穿越"""
                return True

            def render(self):
                # todo
                pass
        return Cross(init_pos, length, color, width, ball_can_pass)

    @staticmethod
    def Agent(mass=1, r=50, position=None, type='agent', color='purple', vis=200, vis_clear=5):
        class GameObj(object):
            """
            Base class for olympic engine objects
            """
            def __init__(self, type, color):
                assert type in OBJECT_TO_IDX, type
                assert color in COLOR_TO_IDX, color
                self.type = type
                self.color = color
                self.name = None
                self.contains = None
                # Initial position of the object
                self.init_pos = None

                # Current position of the object
                self.cur_pos = None

            def can_pass(self):
                """ 是否能穿越"""
                return False

            def can_bounce(self):
                """ 是否能反弹 """
                return False

            def render(self):
                """Draw this object with the given renderer"""
                raise NotImplementedError
        class InternalState(object):
            """
            质量：
            力
            """
            def __init__(self, mass=1, r=30, position=None, vis = None, vis_clear = None):
                self.fatigue = False
                self.energy_cap = 1000
                self.energy = 1000
                self.mass = mass
                # todo: 非正方形视野
                # self.visibility = [100, 100]
                # self.visibility_clear = [2,2] # 清晰度 # 像素大小
                self.visibility = 300 if vis is None else vis
                self.visibility_clear = 12 if vis_clear is None else vis_clear
                self.r = r
                self.position_init = position

                self.alive = True
                self.finished = False

            def reset(self):
                self.fatigue = False
                self.energy = 1000
                # self.visibility = [100, 100]

                self.alive = True
                self.finished = False

            @ property
            def get_property(self):
                return self.energy

            @ property
            def is_fatigue(self):
                if self.energy < 0:
                    self.fatigue = True
                else:
                    self.fatigue = False

                return self.fatigue
        class Agent(GameObj, InternalState):
            def __init__(self, mass=1, r=50, position=None, type='agent', color='purple',
                         vis=300, vis_clear=12):
                # super(Agent, self).__init__(mass=1, r=50, type='agent', color='purple')

                GameObj.__init__(self, type, color)

                InternalState.__init__(self, mass, r, position, vis, vis_clear)

                self.original_color = color

            def can_bounce(self):
                return True

            def reset_color(self):
                self.color = self.original_color

            def render(self):
                # todo
                pass
        return Agent(mass, r, position, type, color, vis, vis_clear)

    @staticmethod
    def engine():
        class OlympicsBase(object):
            def __init__(self, map, seed=None):
                self.VIEW_ITSELF = True
                self.VIEW_BACK = 0.2
                self.seed = seed
                self.set_seed()

                self.action_f = [-100, 200]
                self.action_theta = [-30, 30]

                self.agent_num = 0
                self.agent_list = []
                self.agent_init_pos = []
                self.agent_pos = []
                self.agent_previous_pos = []
                self.agent_v = []
                self.agent_accel = []
                self.agent_theta = []

                self.agent_record = []

                self.show_traj = True
                self.draw_obs = True
                self.print_log = False
                self.print_log2 = False
                self.map_object = []
                self.global_wall_ignore = []
                self.global_circle_ignore = []

                # env hyper
                self.tau = 0.1  # delta t
                self.gamma = 0.98  # v衰减系数
                self.wall_restitution = 0.5
                self.circle_restitution = 0.5

                self.step_cnt = 0
                self.done = False
                self.max_step = 500

                self.energy_recover_rate = 200
                self.speed_cap = 500

                #for debugg
                # self.obs_boundary_init = [[80,350], [180,350],[180,450],[80,450]]
                # self.obs_boundary = self.obs_boundary_init
                self.obs_boundary_init = list()
                self.obs_boundary = self.obs_boundary_init

                self.map = map
                #self.check_valid_map()
                self.generate_map(map)
                self.merge_map()

                self.view_setting = map["view"]
                self.map_num = None
                #self.is_render = True
                self.display_mode = False

                self.reset()
                #self.check_overlap()

            def check_valid_map(self):      #not using due to conflicting with arc center repitition...
                object_init_list = [str(self.map['objects'][i].init_pos) for i in range(len(self.map['objects']))]
                temp = self.map['objects']
                print('len1 = {}, len2 = {}'.format(len(object_init_list),len(set(object_init_list))))
                assert len(object_init_list) == len(set(object_init_list)), print('ERROR: There is repitition in your designed map!')

            def generate_map(self, map):
                for index, item in enumerate(map["agents"]):
                    self.agent_list.append(item)
                    position = item.position_init

                    r = item.r
                    self.agent_init_pos.append(item.position_init)
                    self.agent_num += 1

                    if item.type == 'agent':
                        visibility = item.visibility
                        boundary = self.get_obs_boundaray(position, r, visibility)
                        #print("boundary: ", boundary)
                        self.obs_boundary_init.append(boundary)
                    else:
                        self.obs_boundary_init.append(None)
                    # self.obs_boundary = self.obs_boundary_init
            def merge_map(self):
                point2wall = {}
                for idx, map_item in enumerate(self.map['objects']):
                    if map_item.type != 'wall':
                        continue
                    l1, l2 = tuple(map_item.l1), tuple(map_item.l2)
                    if not map_item.can_pass():
                        if l1 not in point2wall.keys():
                            point2wall[l1] = [idx]
                        else:
                            point2wall[l1].append(idx)
                        if l2 not in point2wall.keys():
                            point2wall[l2] = [idx]
                        else:
                            point2wall[l2].append(idx)
                self.point2wall=point2wall

            def get_obs_boundaray(self, init_position, r, visibility):
                # 默认初始视线水平
                boundary = list()
                x_init, y_init = init_position[0], init_position[1]
                for unit in [[0,1], [1,1], [1,-1], [0,-1]]:
                    if self.VIEW_ITSELF:
                        x = x_init + visibility * unit[0] - self.VIEW_BACK * visibility
                    else:
                        x = x_init + r + visibility * unit[0]

                    y = y_init - visibility * unit[1] / 2
                    boundary.append([x,y])
                return boundary

            @staticmethod
            def create_seed():
                seed = random.randrange(1000)
                return seed

            def set_seed(self, seed=None):
                pass

            def init_state(self):
                self.agent_pos = []
                self.agent_v = []
                self.agent_accel = []
                self.agent_theta = []
                self.agent_record = []
                random_theta = random.uniform(-180, 180)
                for i in range(len(self.agent_list)):
                    self.agent_pos.append(self.agent_init_pos[i])
                    self.agent_previous_pos.append(self.agent_init_pos[i])
                    self.agent_v.append([0,0])
                    self.agent_accel.append([0,0])
                    init_obs = self.view_setting["init_obs"][i] if "init_obs" in self.view_setting else 0#random_theta
                    self.agent_theta.append([init_obs])
                    self.agent_list[i].reset()
                    self.agent_list[i].reset_color()

                    self.agent_record.append([self.agent_init_pos[i]])

            def check_overlap(self):
                #checking the intialisation of agents
                for agent_idx in range(self.agent_num):
                    current_pos = self.agent_pos[agent_idx]
                    r = self.agent_list[agent_idx].r
                    #check wall overlap
                    for object_idx in range(len(self.map['objects'])):
                        if self.map['objects'][object_idx].type == 'wall':
                            #distance from point to line
                            l1,l2 = self.map['objects'][object_idx].init_pos
                            distance = point2line(l1,l2,current_pos)
                            if distance < r:
                                raise ValueError("The position of agent {} overlap with object {}. Please reinitialise...".
                                                 format(agent_idx, object_idx))
                    for rest_idx in range(agent_idx+1, self.agent_num):
                        pos1 = self.agent_pos[agent_idx]
                        r1 = self.agent_list[agent_idx].r
                        pos2 = self.agent_pos[rest_idx]
                        r2 = self.agent_list[rest_idx].r

                        distance = (pos1[0]-pos2[0])**2 + (pos1[1] - pos2[1])**2
                        if distance < (r1+r2)**2:
                            raise ValueError("The position of agent {} overlap with agent {}.".format(agent_idx, rest_idx))


            def reset(self):
                self.set_seed()
                self.init_state()
                self.step_cnt = 0
                self.done = False

                # self.viewer = Viewer(self.view_setting)
                self.display_mode=False

                return self.get_obs()

            def theta_decoder(self):
                if self.theta < 0 or self.theta > 360:
                    self.theta %= 360


            def _circle_collision_response(self, coord1, coord2, v1, v2, m1, m2):
                """
                whether the input represents the new or the old need second thoughts
                :param coord1: position of object 1
                :param coord2: position of object 2
                :param v1:  velocity of object 1
                :param v2:  velocity of object 2
                :param m1: mass of object 1
                :param m2: mass of object 2
                :return:
                """
                n_x = coord1[0] - coord2[0]
                n_y = coord1[1] - coord2[1]

                vdiff_x = (v1[0] - v2[0])
                vdiff_y = (v1[1] - v2[1])

                n_vdiff = n_x * vdiff_x + n_y * vdiff_y
                nn = n_x * n_x + n_y * n_y
                b = n_vdiff/nn

                #object 1
                u1_x = v1[0] - 2*(m2/(m1+m2)) * b * n_x
                u1_y = v1[1] - 2*(m2/(m1+m2)) * b * n_y

                #object 2
                u2_x = v2[0] + 2*(m1/(m1+m2)) * b * n_x
                u2_y = v2[1] + 2*(m1/(m1+m2)) * b * n_y

                return [u1_x*self.circle_restitution, u1_y*self.circle_restitution], \
                       [u2_x*self.circle_restitution, u2_y*self.circle_restitution]


            def CCD_circle_collision(self, old_pos1, old_pos2, old_v1, old_v2, r1, r2, m1, m2, return_t):
                """
                this is the CCD circle collision for the new SetPhysics function
                """

                relative_pos = [old_pos1[0]-old_pos2[0], old_pos1[1]-old_pos2[1]]
                relative_v = [old_v1[0]-old_v2[0], old_v1[1]-old_v2[1]]
                if (relative_v[0]**2+relative_v[1]**2) == 0:
                    return -1

                pos_v = relative_pos[0]*relative_v[0] + relative_pos[1]*relative_v[1]
                K = pos_v/(relative_v[0]**2+relative_v[1]**2)
                l = (relative_pos[0]**2 + relative_pos[1]**2 - (r1+r2)**2)/(relative_v[0]**2+relative_v[1]**2)

                sqrt = (K**2 - l)
                if sqrt <0 and return_t:
                    #print('CCD circle no solution')
                    return -1

                sqrt = math.sqrt(sqrt)
                t1 = -K - sqrt
                t2 = -K + sqrt
                t = min(t1, t2)

                if return_t:
                    return t

                x1,y1 = old_pos1
                x2,y2 = old_pos2
                x1_col = x1 + old_v1[0]*t
                y1_col = y1 + old_v1[1]*t
                x2_col = x2 + old_v2[0]*t
                y2_col = y2 + old_v2[1]*t
                pos_col1, pos_col2 = [x1_col, y1_col], [x2_col, y2_col]

                #handle collision
                v1_col, v2_col = self._circle_collision_response(pos_col1, pos_col2, old_v1, old_v2, m1, m2)      #the position and v at the collision time

                return pos_col1, v1_col, pos_col2, v2_col


            def bounceable_wall_collision_time(self, pos_container, v_container, remaining_t, ignore):

                col_target = None
                col_target_idx = None
                current_idx = None
                current_min_t = remaining_t

                for agent_idx in range(self.agent_num):
                    pos = pos_container[agent_idx]
                    v = v_container[agent_idx]
                    r = self.agent_list[agent_idx].r

                    if v[0] == v[1] == 0:
                        continue

                    for object_idx in range(len(self.map['objects'])):
                        object = self.map['objects'][object_idx]

                        if object.can_pass():     #cross
                            continue
                        if object.ball_can_pass and self.agent_list[agent_idx].type == 'ball':      #for table hockey game
                            continue

                        #check the collision time and the collision target (wall and endpoint collision)
                        temp_t, temp_col_target = object.collision_time(pos = pos, v = v, radius = r, add_info = [agent_idx, object_idx, ignore])

                        if abs(temp_t) < 1e-10:   #the collision time computation has numerical error
                            temp_t = 0


                        #if object_idx == 2:
                        #    print('agent {}: time on wall {}({}) is = {}, current_min_t = {}'.format(
                        #        agent_idx,object_idx,temp_col_target, temp_t, current_min_t))
                        #print('ignore list = ', ignore)

                        if 0<= temp_t < current_min_t:
                            if temp_col_target == 'wall' or temp_col_target == 'arc':
                                check = ([agent_idx, object_idx, temp_t] not in ignore)
                            elif temp_col_target == 'l1' or temp_col_target == 'l2':
                                check = ([agent_idx, getattr(object, temp_col_target), temp_t] not in ignore)
                            else:
                                raise NotImplementedError('bounceable_wall_collision_time error')

                            if check:
                                current_min_t = temp_t
                                col_target = temp_col_target
                                col_target_idx = object_idx
                                current_idx = agent_idx

                return current_min_t, col_target, col_target_idx, current_idx

            def wall_response(self, target_idx, col_target, pos, v, r, t):

                object = self.map['objects'][target_idx]

                #t, v_col = object.collision_time(pos, v, r, compute_response = True, response_col_t = t,restitution = self.restitution)
                col_pos, col_v = object.collision_response(pos, v, r, col_target, t, self.wall_restitution)

                return col_pos, col_v

            def update_other(self, pos_container, v_container, t, already_updated):  # should we also contain a???

                for agent_idx in range(self.agent_num):
                    if agent_idx in already_updated:
                        continue

                    old_pos = pos_container[agent_idx]
                    old_v = v_container[agent_idx]

                    new_x = old_pos[0] + old_v[0] * t
                    new_y = old_pos[1] + old_v[1] * t

                    pos_container[agent_idx] = [new_x, new_y]
                    v_container[agent_idx] = [old_v[0]*self.gamma, old_v[1]*self.gamma]

                return pos_container, v_container

            def update_all(self, pos_container, v_container, t, a):
                for agent_idx in range(self.agent_num):
                    accel_x, accel_y = a[agent_idx]

                    pos_old = pos_container[agent_idx]
                    v_old = v_container[agent_idx]

                    vx, vy = v_old
                    x, y = pos_old
                    x_new = x + vx * t  # update position with t
                    y_new = y + vy * t
                    pos_new = [x_new, y_new]
                    vx_new = self.gamma * vx + accel_x * self.tau  # update v with acceleration
                    vy_new = self.gamma * vy + accel_y * self.tau

                    #if vx_new ** 2 + vy_new ** 2 > self.max_speed_square:
                    #    print('speed limited')
                    #    v_new = [vx, vy]
                    #else:
                    v_new = [vx_new, vy_new]

                    pos_container[agent_idx] = pos_new
                    v_container[agent_idx] = v_new
                return pos_container, v_container


            def circle_collision_time(self, pos_container, v_container, remaining_t, ignore):   #ignore = [[current_idx, target_idx, collision time]]


                #compute collision time between all circle
                current_idx = None
                target_idx = None
                current_min_t = remaining_t
                col_target = None

                for agent_idx in range(self.agent_num):
                    pos1 = pos_container[agent_idx]
                    v1 = v_container[agent_idx]
                    m1 = self.agent_list[agent_idx].mass
                    r1 = self.agent_list[agent_idx].r

                    for rest_idx in range(agent_idx + 1, self.agent_num):
                        pos2 = pos_container[rest_idx]
                        v2 = v_container[rest_idx]
                        m2 = self.agent_list[rest_idx].mass
                        r2 = self.agent_list[rest_idx].r

                        #compute ccd collision time
                        collision_t = self.CCD_circle_collision(pos1, pos2, v1, v2, r1, r2, m1, m2, return_t=True)
                        # print('agent {}, time on circle {} is  = {}, current_min_t = {}'.format(agent_idx,
                        #                                                                         rest_idx,
                        #                                                                         collision_t,
                        #                                                                         remaining_t))
                        # print('ignore list = ', ignore)

                        if 0 <= collision_t < current_min_t:# and [agent_idx, rest_idx, collision_t] not in ignore:
                            current_min_t = collision_t
                            current_idx = agent_idx
                            target_idx = rest_idx
                            col_target = 'circle'

                return current_min_t, col_target, current_idx, target_idx

            def actions_to_accel(self, actions_list):
                """
                Convert action(force) to acceleration; if the action is None, the accel is zero; if the agent is fatigue, the
                force will convert to no acceleration;
                :param actions_list:
                :return:
                """
                a_container = [[] for _ in range(self.agent_num)]
                for agent_idx in range(self.agent_num):
                    action = actions_list[agent_idx]
                    if action is None:
                        accel = [0, 0]
                    else:
                        if self.agent_list[agent_idx].is_fatigue:       #if agent is out of energy, no driving force applies
                            accel = [0,0]
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
                            accel = [accel_x, accel_y]
                            #self.agent_accel[agent_idx] = accel  # update the agent acceleration

                    a_container[agent_idx] = accel
                return a_container

            def _add_wall_ignore(self, collision_wall_target, current_agent_idx, target_wall_idx, ignore_wall, if_global=False):

                if collision_wall_target == 'wall' or collision_wall_target ==  'arc':
                    if if_global:
                        self.global_wall_ignore.append([current_agent_idx, target_wall_idx, 0])
                    else:
                        ignore_wall.append([current_agent_idx, target_wall_idx, 0])

                elif collision_wall_target == 'l1' or collision_wall_target == 'l2':
                    if if_global:
                        self.global_wall_ignore.append(
                            [current_agent_idx, getattr(self.map['objects'][target_wall_idx], collision_wall_target), 0])
                    else:
                        ignore_wall.append(
                            [current_agent_idx, getattr(self.map['objects'][target_wall_idx], collision_wall_target), 0])

                else:
                    raise NotImplementedError("ignore list error")

                if not if_global:
                    return ignore_wall


            # def handle_wall(self, target_wall_idx, col_target, current_agent_idx, col_t,
            #                 pos_container, v_container, remaining_t, ignore_wall_list):
            #
            #     col_pos, col_v = self.wall_response(target_idx=target_wall_idx, col_target = col_target,
            #                                         pos = pos_container[current_agent_idx], v = v_container[current_agent_idx],
            #                                         r = self.agent_list[current_agent_idx], t = col_t)
            #     pos_container[current_agent_idx] = col_pos
            #     v_container[current_agent_idx] = col_v
            #
            #     pos_container, v_container = self.update_other(pos_container=pos_container, v_container=  v_container, t=col_t,
            #                                                    already_updated=[current_agent_idx])
            #     remaining_t -= col_t
            #     if remaining_t <= 1e-14:
            #         self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, None, if_global=True)
            #
            #     ignore_wall_list = self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, ignore_wall_list)
            #
            #     return pos_container, v_container, remaining_t, ignore_wall_list
            def handle_wall(self, target_wall_idx, col_target, current_agent_idx, col_t,
                            pos_container, v_container, remaining_t, ignore_wall_list):

                col_pos, col_v = self.wall_response(target_idx=target_wall_idx, col_target = col_target,
                                                    pos = pos_container[current_agent_idx], v = v_container[current_agent_idx],
                                                    r = self.agent_list[current_agent_idx], t = col_t)
                pos_container[current_agent_idx] = col_pos
                v_container[current_agent_idx] = col_v

                pos_container, v_container = self.update_other(pos_container=pos_container, v_container=  v_container, t=col_t,
                                                               already_updated=[current_agent_idx])
                remaining_t -= col_t
                if remaining_t <= 1e-14:

                    if col_target == 'wall':
                        self._add_wall_ignore('wall', current_agent_idx, target_wall_idx, ignore_wall_list,if_global=True)
                        self._add_wall_ignore('l1', current_agent_idx, target_wall_idx, ignore_wall_list,if_global=True)
                        self._add_wall_ignore('l2', current_agent_idx, target_wall_idx, ignore_wall_list,if_global=True)
                    elif col_target == 'l2' or col_target == 'l1':
                        self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx,
                                              ignore_wall_list, if_global=True)
                        wall_endpoint = getattr(self.map['objects'][target_wall_idx], col_target)
                        connected_wall = self.point2wall[tuple(wall_endpoint)]
                        for idx in connected_wall:
                            self._add_wall_ignore('wall', current_agent_idx, idx, ignore_wall_list, if_global=True)
                    else:
                        self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, None, if_global=True)         #collision of arc

                if col_target == 'wall':
                    ignore_wall_list = self._add_wall_ignore('wall', current_agent_idx, target_wall_idx, ignore_wall_list)

                    ignore_wall_list = self._add_wall_ignore('l1', current_agent_idx, target_wall_idx, ignore_wall_list)
                    ignore_wall_list = self._add_wall_ignore('l2', current_agent_idx, target_wall_idx, ignore_wall_list)
                elif col_target == 'l2' or col_target == 'l1':
                    ignore_wall_list = self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, ignore_wall_list)
                    wall_endpoint = getattr(self.map['objects'][target_wall_idx], col_target)
                    connected_wall = self.point2wall[tuple(wall_endpoint)]
                    for idx in connected_wall:
                        ignore_wall_list = self._add_wall_ignore('wall', current_agent_idx, idx, ignore_wall_list)
                else:
                    ignore_wall_list = self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, ignore_wall_list)


                return pos_container, v_container, remaining_t, ignore_wall_list

            def handle_circle(self, target_circle_idx, col_target, current_circle_idx, col_t,
                              pos_container, v_container, remaining_t, ignore_circle_list):

                pos_col1, v1_col, pos_col2, v2_col = \
                    self.CCD_circle_collision(pos_container[current_circle_idx], pos_container[target_circle_idx],
                                              v_container[current_circle_idx], v_container[target_circle_idx],
                                              self.agent_list[current_circle_idx].r, self.agent_list[target_circle_idx].r,
                                              self.agent_list[current_circle_idx].mass, self.agent_list[target_circle_idx].mass,
                                              return_t = False)

                pos_container[current_circle_idx], pos_container[target_circle_idx] = pos_col1, pos_col2
                v_container[current_circle_idx], v_container[target_circle_idx] = v1_col, v2_col

                pos_container, v_container = self.update_other(pos_container, v_container, col_t,
                                                               [current_circle_idx, target_circle_idx])
                remaining_t -= col_t
                if remaining_t <= 1e-14:
                    self.global_circle_ignore.append([current_circle_idx, target_circle_idx, 0.0])  # adding instead of defining
                ignore_circle_list.append([current_circle_idx, target_circle_idx, 0.0])

                return pos_container, v_container, remaining_t, ignore_circle_list



            def stepPhysics(self, actions_list, step = None):

                assert len(actions_list) == self.agent_num, print("The number of action needs to match with the number of agents!")
                self.actions_list = actions_list

                #current pos and v
                temp_pos_container = [self.agent_pos[i] for i in range(self.agent_num)]
                temp_v_container = [self.agent_v[i] for i in range(self.agent_num)]
                temp_a_container = self.actions_to_accel(actions_list)
                self.agent_accel = temp_a_container

                remaining_t = self.tau
                ignore_wall = copy.copy(self.global_wall_ignore)
                ignore_circle = copy.copy(self.global_circle_ignore)
                self.global_wall_ignore, self.global_circle_ignore = [], []   #only inherit once


                while True:
                    if self.print_log:
                        print('Remaining time = ', remaining_t)
                        print('The pos = {}, the v = {}'.format(temp_pos_container, temp_v_container))

                    earliest_wall_col_t, collision_wall_target, target_wall_idx, current_agent_idx = \
                        self.bounceable_wall_collision_time(temp_pos_container, temp_v_container, remaining_t, ignore_wall)    #collision detection with walls

                    earliest_circle_col_t, collision_circle_target, current_circle_idx, target_circle_idx = \
                        self.circle_collision_time(temp_pos_container, temp_v_container, remaining_t, ignore_circle)

                    if self.print_log:
                        print('Wall t = {}, collide = {}, agent_idx = {}, wall_idx = {}'.format(
                            earliest_wall_col_t, collision_wall_target, current_agent_idx, target_wall_idx))
                        print('Circle t = {}, collide = {}, agent_idx = {}, target_idx = {}'.format(
                            earliest_circle_col_t, collision_circle_target, current_circle_idx, target_circle_idx))


                    if collision_wall_target is not None and collision_circle_target is None:
                        if self.print_log:
                            print('HIT THE WALL!')

                        temp_pos_container, temp_v_container, remaining_t, ignore_wall = \
                            self.handle_wall(target_wall_idx, collision_wall_target, current_agent_idx, earliest_wall_col_t,
                                             temp_pos_container, temp_v_container, remaining_t, ignore_wall)

                    elif collision_wall_target is None and collision_circle_target == 'circle':
                        if self.print_log:
                            print('HIT THE BALL!')

                        temp_pos_container, temp_v_container, remaining_t, ignore_circle = \
                            self.handle_circle(target_circle_idx, collision_circle_target, current_circle_idx,
                                               earliest_circle_col_t, temp_pos_container, temp_v_container, remaining_t,
                                               ignore_circle)

                    elif collision_wall_target is not None and collision_circle_target == 'circle':
                        if self.print_log:
                            print('HIT BOTH!')

                        if earliest_wall_col_t < earliest_circle_col_t:
                            if self.print_log:
                                print('PROCESS WALL FIRST!')

                            temp_pos_container, temp_v_container, remaining_t, ignore_wall = \
                                self.handle_wall(target_wall_idx, collision_wall_target, current_agent_idx, earliest_wall_col_t,
                                                 temp_pos_container, temp_v_container, remaining_t, ignore_wall)

                        elif earliest_wall_col_t >= earliest_circle_col_t:
                            if self.print_log:
                                print('PROCESS CIRCLE FIRST!')

                            temp_pos_container, temp_v_container, remaining_t, ignore_circle = \
                                self.handle_circle(target_circle_idx, collision_circle_target, current_circle_idx,
                                                   earliest_circle_col_t, temp_pos_container, temp_v_container, remaining_t,
                                                   ignore_circle)

                        else:
                            raise NotImplementedError("collision error")

                    else:   #no collision within this time interval
                        if self.print_log:
                            print('NO COLLISION!')
                        temp_pos_container, temp_v_container = self.update_all(temp_pos_container, temp_v_container, remaining_t, temp_a_container)
                        break   #when no collision occurs, break the collision detection loop

                self.agent_pos = temp_pos_container
                self.agent_v = temp_v_container

                #if self.is_render:
                #    debug('Step: ' + str(step), x = 30)
                #    pos = [(float('%.1f' % i[0]), float('%.1f' % i[1])) for i in self.agent_pos]
                #    debug('Agent pos = ' + str(pos),x = 120)
                #    speed = [ float('%.1f' % (math.sqrt(i[0]**2 + i[1]**2))) for i in self.agent_v  ]
                #    debug('Agent speed = ' + str(speed), x = 420)

            def cross_detect2(self):
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
                                agent.finished = True       #when agent has crossed the finished line
                                agent.alive = False         #kill the agent when finishing the task

            def cross_detect(self, previous_pos, new_pos):

                for object_idx in range(len(self.map['objects'])):
                    object = self.map['objects'][object_idx]
                    if object.can_pass() and object.color == 'red':
                        l1,l2 = object.init_pos         #locate the pos of Final
                        final = object

                for agent_idx in range(self.agent_num):
                    agent = self.agent_list[agent_idx]
                    agent_pre_pos, agent_new_pos = previous_pos[agent_idx], new_pos[agent_idx]

                    if line_intersect(line1 = [l1, l2], line2 = [agent_pre_pos, agent_new_pos]):
                        agent.color = 'red'
                        agent.finished = True
                        agent.alive = False

                    if (point2line(l1, l2, agent_new_pos) <= self.agent_list[agent_idx].r) and final.check_on_line(closest_point(l1,l2,agent_new_pos)):
                        #if the center of circle to the line has distance less or equal to the radius, and the closest point is on the line, then cross the final
                        agent.color = 'red'
                        agent.finished = True
                        agent.alive = False


            def get_obs(self):
                """
                POMDP: partial observation
                """



                # this is for debug
                self.obs_boundary = list()


                obs_list = list()

                for agent_idx, agent in enumerate(self.agent_list):
                    if self.agent_list[agent_idx].type == 'ball':
                        self.obs_boundary.append(None)
                        obs_list.append(None)
                        continue

                    # time_s = time.time()
                    theta_copy = self.agent_theta[agent_idx][0]
                    agent_pos = self.agent_pos
                    agent_x, agent_y = agent_pos[agent_idx][0], agent_pos[agent_idx][1]
                    theta = theta_copy
                    position_init = agent.position_init

                    visibility = self.agent_list[agent_idx].visibility
                    v_clear = self.agent_list[agent_idx].visibility_clear
                    # obs_map = np.zeros((visibility[0], visibility[1]))
                    # obs_weight,obs_height = int(visibility[0]/v_clear[0]),int(visibility[1]/v_clear[1])
                    obs_size = int(visibility / v_clear)
                    view_back = visibility*self.VIEW_BACK
                    # update_obs_boundary()
                    agent_current_boundary = list()
                    for b in self.obs_boundary_init[agent_idx]:
                        m = b[0]
                        n = b[1]
                        # print("check b orig: ", b_x, b_y)
                        vec_oo_ = (-agent_x, agent_y)
                        vec = (-position_init[0], position_init[1])
                        vec_o_a = (m, -n)
                        # vec_oa = (vec_oo_[0]+vec_o_a[0], vec_oo_[1]+vec_o_a[1])
                        vec_oa = (vec[0] + vec_o_a[0], vec[1] + vec_o_a[1])
                        b_x_ = vec_oa[0]
                        b_y_ = vec_oa[1]
                        # print("check b: ", b_x_, b_y_)
                        x_new, y_new = rotate2(b_x_, b_y_, theta)
                        # print("check x_new: ", x_new, y_new)
                        # x_new_ = x_new + agent_x
                        # y_new_ = -y_new + agent_y
                        x_new_ =  x_new - vec_oo_[0]
                        y_new_ =  y_new - vec_oo_[1]
                        agent_current_boundary.append([x_new_, -y_new_])
                    self.obs_boundary.append(agent_current_boundary)

                    #compute center of view, need to fix for non-view-self
                    view_center_x = agent_x + (visibility/2-view_back/2)*math.cos(theta*math.pi/180)      #start from agent x,y
                    view_center_y = agent_y + (visibility/2-view_back/2)*math.sin(theta*math.pi/180)
                    view_center = [view_center_x, view_center_y]
                    view_R = visibility*math.sqrt(2)/2
                    line_consider = []

                    # rotate coord
                    for index_m, item in enumerate(self.map["objects"]):
                        if (item.type == "wall") or (item.type == "cross"):
                            closest_dist = distance_to_line(item.init_pos[0], item.init_pos[1], view_center)
                            if closest_dist <= view_R:
                                line_consider.append(item)

                        elif item.type == "arc":
                            pos = item.center
                            item.cur_pos = list()
                            vec_o_d = (pos[0], -pos[1])
                            vec_oo_ = (-agent_x, agent_y)
                            vec_od = (vec_o_d[0] + vec_oo_[0], vec_o_d[1] + vec_oo_[1])
                            item.cur_pos.append([vec_od[0], vec_od[1]])

                    # distance to view center
                    if self.VIEW_ITSELF:
                        vec_oc = (visibility/2-view_back/2, 0)
                    else:
                        vec_oc = (agent.r+visibility/2, 0)
                    c_x = vec_oc[0]
                    c_y = vec_oc[1]
                    c_x_, c_y_ = rotate2(c_x, c_y, theta)
                    vec_oc_ = (c_x_, c_y_)

                    map_objects = list()
                    map_deduced = dict()
                    map_deduced["objects"] = list()
                    for c in self.map["objects"]:
                        if (c.type == "wall") or (c.type == "cross"):
                            pass

                        elif c.type == "arc":
                            distance = distance_2points([c.cur_pos[0][0]-vec_oc_[0],c.cur_pos[0][1]-vec_oc_[1]])
                            if distance <= visibility/2 * 1.415 + c.R:
                                map_deduced["objects"].append(c)
                                map_objects.append(c)
                        else:
                            raise ValueError("No such object type- {}. Please check scenario.json".
                                             format(c.type))

                    map_deduced["agents"] = list()

                    # current agent it self
                    agent_self = self.agent_list[agent_idx]
                    agent_self.to_another_agent = []
                    agent_self.to_another_agent_rotated = []
                    temp_idx = 0
                    #for a_i, a_other in enumerate(self.map["agents"]):
                    for a_i, a_other in enumerate(self.agent_list):
                        if a_i == agent_idx:
                            continue
                        else:
                            vec_o_b = (self.agent_pos[a_i][0], -self.agent_pos[a_i][1])
                            vec_oo_ = (-agent_x, agent_y)
                            vec_ob = (vec_o_b[0]+vec_oo_[0], vec_o_b[1]+vec_oo_[1])
                            vec_bc_ = (vec_oc_[0]-vec_ob[0], vec_oc_[1]-vec_ob[1])
                            # agent_self.to_another_agent = vec_ob
                            distance = math.sqrt(vec_bc_[0]**2 + vec_bc_[1]**2)
                            threshold = self.agent_list[agent_idx].visibility * 1.415 / 2 + a_other.r # 默认视线为正方形
                            if distance <= threshold:
                                map_deduced["agents"].append(a_i) # todo
                                a_other.temp_idx = temp_idx
                                map_objects.append(a_other)
                                agent_self.to_another_agent.append(vec_ob)
                                temp_idx += 1

                    obs_map = np.zeros((obs_size,obs_size))
                    for obj in map_deduced["objects"]:
                        if (obj.type == "wall") or (obj.type == "cross"):
                            points_pos = obj.cur_pos
                            obj.cur_pos_rotated = list()
                            for pos in points_pos:
                                pos_x = pos[0]
                                pos_y = pos[1]
                                theta_obj = - theta
                                pos_x_, pos_y_ = rotate2(pos_x, pos_y, theta_obj)
                                obj.cur_pos_rotated.append([pos_x_, pos_y_])
                        elif obj.type == "arc":
                            pos = obj.cur_pos
                            obj.cur_pos_rotated = list()
                            pos_x = pos[0][0]
                            pos_y = pos[0][1]
                            theta_obj = - theta
                            pos_x_, pos_y_ = rotate2(pos_x, pos_y, theta_obj)
                            obj.cur_pos_rotated.append([pos_x_, pos_y_])

                    for id, obj in enumerate(map_deduced["agents"]):
                        vec_ob = agent_self.to_another_agent[id]  # todo: 现在只有两个agent
                        theta_obj = - theta
                        x, y = rotate2(vec_ob[0], vec_ob[1], theta_obj)
                        agent_self.to_another_agent_rotated.append((x,y))

                    # now start drawing line
                    for obj in line_consider:
                        if obj.type == 'wall' or obj.type == 'cross':
                            current_pos = obj.init_pos
                            obj.rotate_pos = []
                            for end_point in current_pos:
                                obj.rotate_pos.append(point_rotate([agent_x, agent_y], end_point, theta))

                            # compute the intersection point
                            intersect_p = []
                            rotate_boundary = [[[0-view_back, -visibility / 2], [0-view_back, visibility / 2]],
                                               [[0-view_back, visibility / 2], [visibility-view_back, visibility / 2]],
                                               [[visibility-view_back, visibility / 2], [visibility-view_back, -visibility / 2]],
                                               [[visibility-view_back, -visibility / 2], [0-view_back, -visibility / 2]]]

                            # obs_rotate_boundary = []              #debug rotate boundary
                            # for line in self.obs_boundary:
                            #     rotate_bound = [point_rotate([agent_x, agent_y], i, theta) for i in line]
                            #     obs_rotate_boundary.append(rotate_bound)

                            for line in rotate_boundary:
                                _intersect_p = line_intersect(line1=line, line2=obj.rotate_pos, return_p=True)
                                if _intersect_p:
                                    intersect_p.append(_intersect_p)

                            intersect_p = [tuple(i) for i in intersect_p]
                            intersect_p = list(set(intersect_p))            #ensure no point repetition


                            draw_line = []
                            if len(intersect_p) == 0:
                                point_1_in_view=  0 < obj.rotate_pos[0][0]+view_back < visibility and abs(obj.rotate_pos[0][1]) < visibility / 2
                                point_2_in_view = 0 < obj.rotate_pos[1][0]+view_back < visibility and abs(obj.rotate_pos[1][1]) < visibility / 2

                                if point_1_in_view and point_2_in_view:
                                    draw_line.append(obj.rotate_pos[0])
                                    draw_line.append(obj.rotate_pos[1])
                                elif not point_1_in_view and not point_2_in_view:
                                    continue
                                else:
                                    raise NotImplementedError

                            elif len(intersect_p) == 1:

                                draw_line.append(intersect_p[0])

                                if 0 < obj.rotate_pos[0][0]+view_back < visibility and abs(
                                        obj.rotate_pos[0][1]) < visibility / 2:
                                    draw_line.append(obj.rotate_pos[0])
                                elif 0 < obj.rotate_pos[1][0]+view_back < visibility and abs(
                                        obj.rotate_pos[1][1]) < visibility / 2:
                                    draw_line.append(obj.rotate_pos[1])
                                else:
                                    # only one point in the view
                                    pass

                            elif len(intersect_p) == 2:

                                draw_line.append(intersect_p[0])
                                draw_line.append(intersect_p[1])

                            elif len(intersect_p) == 3:     #if line aligns with boundary
                                continue
                            else:
                                raise ValueError('ERROR: multiple intersection points in DDA')

                            obs_map = DDA_line(obs_map, draw_line, visibility, v_clear,
                                               value=COLOR_TO_IDX[obj.color], view_back=view_back)

                        else:
                            raise NotImplementedError


                    #for component in map_objects:
                    if len(list(reversed(map_objects))) == 0 and self.VIEW_ITSELF:   #if no object in the view, plot the agent itself
                        for i in range(obs_size):
                            x = visibility - v_clear * i - v_clear / 2 - view_back
                            for j in range(obs_size):
                                y = visibility/2 - v_clear*j - v_clear /2
                                point = (x, y)

                                self_center = [0, 0]
                                dis_to_itself = math.sqrt((point[0] - self_center[0]) ** 2 + (point[1] - self_center[1]) ** 2)
                                if dis_to_itself <= self.agent_list[agent_idx].r:
                                    obs_map[i][j] = COLOR_TO_IDX[self.agent_list[agent_idx].color]


                    for component in list(reversed(map_objects)):           #reverse to consider agent first, then wall
                        for i in range(obs_size):
                            if self.VIEW_ITSELF:
                                x = visibility - v_clear*i - v_clear/2 - view_back
                            else:
                                x = agent.r + visibility - v_clear*i - v_clear / 2

                            for j in range(obs_size):
                                y = visibility/2 - v_clear*j - v_clear /2
                                point = (x, y)

                                if self.VIEW_ITSELF:
                                    #plot the agnet it self
                                    self_center = [0, 0]
                                    dis_to_itself = math.sqrt((point[0]-self_center[0])**2 + (point[1]-self_center[1])**2)
                                    if dis_to_itself <= self.agent_list[agent_idx].r:
                                        obs_map[i][j] = COLOR_TO_IDX[self.agent_list[agent_idx].color]


                                if obs_map[i][j] > 0 and (component.type != 'agent' and component.type != 'ball'):           #when there is already object on this pixel
                                    continue
                                else:
                                    if (component.type == "wall") or (component.type == "cross"):
                                        distance = abs(get_distance(component.cur_pos_rotated, point, component.length,
                                                                    pixel=True))
                                        if distance <= v_clear :  # 距离小于等于1个像素点长度
                                            obs_map[i][j] = COLOR_TO_IDX[component.color]
                                    elif component.type == "agent" or component.type == "ball":
                                        idx = component.temp_idx
                                        vec_bc_ = (x-agent_self.to_another_agent_rotated[idx][0], y-agent_self.to_another_agent_rotated[idx][1])
                                        distance = math.sqrt(vec_bc_[0] ** 2 + vec_bc_[1] ** 2)
                                        if distance <= component.r:
                                            obs_map[i][j] = COLOR_TO_IDX[component.color]
                                    elif component.type == "arc":
                                        radius = component.R
                                        x_2center, y_2center = x-component.cur_pos_rotated[0][0], y-component.cur_pos_rotated[0][1]
                                        theta_pixel = theta
                                        pos_x_2center, pos_y_2center = rotate2(x_2center,y_2center,theta_pixel)
                                        angle = math.atan2(pos_y_2center, pos_x_2center)
                                        start_radian, end_radian = component.start_radian, component.end_radian
                                        if get_obs_check_radian(start_radian,end_radian,angle):
                                            # if (angle >= start_radian) and (angle <= end_radian):
                                            vec = [x - component.cur_pos_rotated[0][0], y - component.cur_pos_rotated[0][1]]
                                            distance = distance_2points(vec)
                                            if (distance <= radius + v_clear/2) and (distance >= radius - v_clear/2):
                                                obs_map[i][j] = COLOR_TO_IDX[component.color]

                    obs_list.append(obs_map)
                self.obs_list = obs_list


                return obs_list

            def change_inner_state(self):

                for agent_idx in range(self.agent_num):
                    if self.agent_list[agent_idx].type == 'ball':
                        continue

                    if self.agent_list[agent_idx].energy < 0:       #once fatigue, the agent died and lose control
                        remaining_energy = -1
                    else:

                        #previous_pos = self.agent_previous_pos[agent_idx]
                        #current_pos = self.agent_pos[agent_idx]

                        #moved = math.sqrt((previous_pos[0] - current_pos[0])**2 + (previous_pos[1] - current_pos[1])**2)


                        #force_idx = 0 if self.actions_list[agent_idx] is None else self.actions_list[agent_idx][0]
                        force = self.agent_list[agent_idx].mass * math.sqrt(self.agent_accel[agent_idx][0]**2 +
                                                                            self.agent_accel[agent_idx][1]**2)
                        v = math.sqrt(self.agent_v[agent_idx][0]**2 + self.agent_v[agent_idx][1]**2)
                        consume_rate = force*v/50
                        consume_rate -= self.energy_recover_rate

                        remaining_energy = self.agent_list[agent_idx].energy - consume_rate*self.tau
                        #force = force_idx

                        #energy_used = force * moved/1000          #F*s

                        #remaining_energy = self.agent_list[agent_idx].energy - energy_used + self.energy_recover_rate*self.tau
                        if remaining_energy <0:
                            remaining_energy = -1
                        elif remaining_energy > self.agent_list[agent_idx].energy_cap:
                            remaining_energy = self.agent_list[agent_idx].energy_cap
                        else:
                            pass

                    #print('remaining energy = ', remaining_energy)
                    self.agent_list[agent_idx].energy = remaining_energy
                    #self.agent_list[agent_idx].energy -= (energy_used - self.energy_recover_rate*self.tau)

            def speed_limit(self):

                for agent_idx in range(self.agent_num):
                    current_v = self.agent_v[agent_idx]
                    current_speed = math.sqrt(current_v[0]**2 + current_v[1]**2)

                    if current_speed > self.speed_cap:
                        factor = self.speed_cap/current_speed
                        cap_v = [current_v[0] * factor, current_v[1] * factor]
                        self.agent_v[agent_idx] = cap_v

                        #print('OVER-SPEED!!!')
        return OlympicsBase

    @staticmethod
    def Running_competition(Gamemap, map_id=None, seed=None, vis=None, vis_clear=None, agent1_color='purple', agent2_color='green'):
        Base = EnvWrapped.engine()
        class Running_competition(Base):
            def __init__(self, Gamemap, map_id = None, seed = None, vis = None, vis_clear=None, agent1_color = 'purple', agent2_color = 'green'):
                # self.minimap_mode = map['obs_cfg'].get('minimap', False)

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



                super(Running_competition, self).__init__(Gamemap, seed)

                self.game_name = 'running-competition'

                self.original_tau = 0.1
                self.original_gamma = 0.98
                self.wall_restitution = 0.3
                self.circle_restitution = 0.9
                self.max_step =500
                self.energy_recover_rate = 200
                self.speed_cap = 500
                self.faster = 1

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

            # @staticmethod
            # def reset_map(meta_map, map_id, vis=None, vis_clear=None, agent1_color = 'purple', agent2_color = 'green'):
            #     return Running_competition(meta_map, map_id, vis=vis, vis_clear = vis_clear, agent1_color=agent1_color, agent2_color=agent2_color)

            # @staticmethod
            # def choose_a_map(idx=None):
            #     if idx is None:
            #         idx = random.randint(1,4)
            #     MapStats = create_scenario("map"+str(idx), file_path=  maps_path)
            #     return MapStats, idx

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

                self.stepPhysics(actions_list, self.step_cnt)
                #print('stepPhysics time = ', time2 - time1)
                self.speed_limit()

                self.cross_detect(previous_pos, self.agent_pos)

                self.step_cnt += 1
                step_reward = self.get_reward()
                done = self.is_terminal()

                obs_next = self.get_obs()
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
        return Running_competition(Gamemap, map_id, seed, vis, vis_clear, agent1_color, agent2_color)



    def reset(self):
        self.episode_len = 0
        self.cur_obs = self.env.reset()[self.ctrl_agent_index].flatten()
        self.cur_obs = self.cur_obs.astype(int)

        return self.cur_obs

    def step(self, action):
        self.episode_len += 1
        input_action = [[],[]]
        input_action[self.ctrl_agent_index] = action
        input_action[1-self.ctrl_agent_index] = [0,0]

        next_obs, reward, done, _ = self.env.step(input_action)
        self.cur_obs = next_obs[self.ctrl_agent_index].flatten()

        return self.cur_obs.astype(int), reward[self.ctrl_agent_index], done, {}

env = EnvWrapped('')





# trainer = PPOTrainer(
#     config={
#         "env": EnvWrapped,
#         "env_config": "",
#         "num_workers": 1
#     }
# )
#
# for i in range(5):
#     results = trainer.train()
#     print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
#
# #
# class ParrotEnv(gym.Env):
#     """Environment in which an agent must learn to repeat the seen observations.
#
#     Observations are float numbers indicating the to-be-repeated values,
#     e.g. -1.0, 5.1, or 3.2.
#
#     The action space is always the same as the observation space.
#
#     Rewards are r=-abs(observation - action), for all steps.
#     """
#
#     def __init__(self, config):
#         # Make the space (for actions and observations) configurable.
#         self.action_space = config.get(
#             "parrot_shriek_range", gym.spaces.Box(-1.0, 1.0, shape=(1, )))
#         # Since actions should repeat observations, their spaces must be the
#         # same.
#         self.observation_space = self.action_space
#         self.cur_obs = None
#         self.episode_len = 0
#
#     def reset(self):
#         """Resets the episode and returns the initial observation of the new one.
#         """
#         # Reset the episode len.
#         self.episode_len = 0
#         # Sample a random number from our observation space.
#         self.cur_obs = self.observation_space.sample()
#         # Return initial observation.
#         return self.cur_obs
#
#     def step(self, action):
#         """Takes a single step in the episode given `action`
#
#         Returns:
#             New observation, reward, done-flag, info-dict (empty).
#         """
#         # Set `done` flag after 10 steps.
#         print('action = ', action)
#         self.episode_len += 1
#         done = self.episode_len >= 10
#         # r = -abs(obs - action)
#         reward = -sum(abs(self.cur_obs - action))
#         # Set a new observation (random sample).
#         self.cur_obs = self.observation_space.sample()
#         return self.cur_obs, reward, done, {}
#
#
# # Create an RLlib Trainer instance to learn how to act in the above
# # environment.
# trainer = PPOTrainer(
#     config={
#         # Env class to use (here: our gym.Env sub-class from above).
#         "env": ParrotEnv,
#         # Config dict to be passed to our custom env's constructor.
#         "env_config": {
#             "parrot_shriek_range": gym.spaces.Box(-5.0, 5.0, (1, ))
#         },
#         # Parallelize environment rollouts.
#         "num_workers": 2,
#     })
#
# # Train for n iterations and report results (mean episode rewards).
# # Since we have to guess 10 times and the optimal reward is 0.0
# # (exact match between observation and action value),
# # we can expect to reach an optimal episode reward of 0.0.
# for i in range(50):
#     results = trainer.train()
#     print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
#
#
#
