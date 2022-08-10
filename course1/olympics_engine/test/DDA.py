import math
import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
print(base_path)


from core import OlympicsBase
from objects import *
from tools.func import *
import time
import numpy as np

gamemap = {'objects':[], 'agents':[]}

# gamemap['objects'].append(Wall(init_pos = [[250, 600], [300, 0]], color = 'black'))
# gamemap['objects'].append(Wall(init_pos = [[150, 100], [500, 60]], color = 'black'))
# gamemap['objects'].append(Wall(init_pos = [[150, 100], [400, 20]], color = 'black'))
gamemap['objects'].append(Wall(init_pos = [[250, 100], [300, 150]], color = 'black'))



gamemap['objects'].append(Arc(init_pos = [100,150, 400, 400], start_radian = 0, end_radian = -100,
                              passable=True, color = 'red', collision_mode = 3))# gamemap['objects'].append(Wall(init_pos = [[200, 10], [100, 200]], color = 'black'))
# gamemap['objects'].append(Wall(init_pos = [[400, 10], [500, 200]], color = 'black'))
# gamemap['objects'].append(Wall(init_pos = [[500, 200], [100, 200]], color = 'black'))

gamemap['agents'].append(Agent(position = [290,100], mass=1, r=15, color='purple', vis=200, vis_clear=5))



gamemap['view'] = {'width': 600, 'height':600, 'edge': 50, "init_obs": [-1]}


def point_rotate(center, point, theta):
    px = point[0] - center[0]
    py = point[1] - center[1]

    nx = [math.cos(theta * math.pi / 180), math.sin(theta * math.pi / 180)]
    ny = [-math.sin(theta * math.pi / 180), math.cos(theta * math.pi / 180)]
    new_x = px * nx[0] + py * nx[1]
    new_y = px * ny[0] + py * ny[1]
    return [new_x, new_y]

def DDA_line(matrix, draw_line, vis, vis_clear, value):
    size = int(vis/vis_clear)
    assert matrix.shape[0] == size
    if len(draw_line) == 1:
        point1 = draw_line[0]
        x1, y1 = point1
        y1 += vis/2
        x1 /= vis_clear
        y1 /= vis_clear

        x = x1-0.5
        y = y1-0.5
        matrix[size-int(x)][int(y)] = 1
        return matrix

    elif len(draw_line) == 2:
        point1, point2 = draw_line
    else:
        raise NotImplementedError

    x1,y1 = point1
    y1 += vis/2
    x1 /= vis_clear
    y1 /= vis_clear
    x2, y2 = point2
    y2 += vis/2
    x2 /= vis_clear
    y2 /= vis_clear

    dx = x2-x1
    dy = y2-y1

    if abs(dx) > abs(dy):
        steps = abs(dx)
    else:
        steps = abs(dy)

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



def arc_line_intersect(line, arc_center, arc_start_radian, arc_end_radian, arc_R):
    """
    compute intersection between arc and line
    """
    assert -math.pi<=arc_start_radian <= math.pi, print(arc_start_radian)
    assert -math.pi<=arc_end_radian <= math.pi, print(arc_end_radian)


    l1, l2 = line
    A = l1[1] - l2[1]
    B = l2[0] - l1[0]
    C = (l1[1] - l2[1]) * l1[0] + (l2[0] - l1[0]) * l1[1]       #Ax + By = C

    if B == 0:
        x = C/A
        sqrt = arc_R**2-(x-arc_center[0])**2
        # assert sqrt >=0
        if sqrt < 0:
            return []
        sqrt = math.sqrt(sqrt)
        y_plus = sqrt + arc_center[1]
        y_neg = -sqrt + arc_center[1]
        x_plus, x_neg = x, x
    else:

        K = arc_center[0]**2 + (C**2)/(B**2) -2*arc_center[1]*C/B + arc_center[1]**2 - arc_R**2
        gamma = -arc_center[0]-A*C/(B**2) + arc_center[1]*A/B
        M = 1 + A**2/B**2
        sqrt = -K/M + (gamma/M)**2
        # assert sqrt >= 0
        if sqrt <0:
            return []
        sqrt = math.sqrt(sqrt)
        x_plus = sqrt - gamma/M
        y_plus = C/B - A/B*x_plus

        x_neg = -sqrt - gamma/M
        y_neg = C/B - A/B*x_neg

        assert abs(A*x_plus+B*y_plus-C)<1e-6
        assert abs(A*x_neg+B*y_neg-C)<1e-6
        assert abs(math.sqrt((x_plus-arc_center[0])**2 + (y_plus-arc_center[1])**2)-arc_R) <1e-6
        assert abs(math.sqrt((x_neg-arc_center[0])**2 + (y_neg-arc_center[1])**2)-arc_R) <1e-6


    valid_point = []
    #check on the line
    if check_on_line(point=[x_plus, y_plus], A=A, B=B, C=C, l1=l1, l2=l2):
        valid_point.append([x_plus, y_plus])
    if check_on_line(point=[x_neg, y_neg], A=A, B=B, C=C, l1=l1, l2=l2):
        valid_point.append([x_neg, y_neg])


    #check radian
    radian_valid_point = []
    for p in valid_point:
        p_radian = math.atan2(arc_center[1]-p[1], p[0]-arc_center[0])       #because pygame y axis is reversed
        if get_obs_check_radian(start_radian=arc_start_radian, end_radian=arc_end_radian, angle=p_radian):
            radian_valid_point.append({"point": p, "rad": p_radian})

    return radian_valid_point


def check_on_line(point, A, B, C, l1, l2):
    temp = A * point[0] + B * point[1]
    #if temp == self.C:     #on the line or at least the line extension
    if abs(temp-C) <= 1e-6:
        if not ((min(l1[0], l2[0]) <= point[0] <= max(l1[0], l2[0])) and (
                min(l1[1], l2[1]) <= point[1] <= max(l1[1], l2[1]))):
            return False        #not on the line segment
        else:
            return True
    else:
        raise NotImplementedError

    return False


# print('arc line intersection', arc_line_intersect(line=[[250, 600], [300, 0]], arc_center=[300, 350],
#                                                   arc_start_radian=0*math.pi/180, arc_end_radian=-100*math.pi/180, arc_R=200))


def bresenham_arc(matrix, draw_arc, vis, vis_clear, value):
    """
    draw_arc contains starting radian and point, end radian and point,center position, arc starting radian, arc R
    """
    size = int(vis/vis_clear)
    assert matrix.shape[0] == size


    # start_point = draw_arc.get('start point')
    start_radian = draw_arc.get('start radian')
    # end_point = draw_arc.get('end_point', None)
    end_radian = draw_arc.get('end radian', None)
    # arc_start_radian = draw_arc.get('arc start radian')
    # arc_end_radian = draw_arc.get('arc end radian')
    arc_center = draw_arc.get('arc center')     #rotate center
    arc_R = draw_arc.get('arc R')

    # x1, y1 = start_point
    # y1 += vis / 2
    # x1 /= vis_clear
    # y1 /= vis_clear
    cx, cy = arc_center
    cx /= vis_clear
    cy /= vis_clear

    quarter = [[0,math.pi/4], [math.pi/4, math.pi/2], [math.pi/2, math.pi*3/4], [math.pi*3/4, math.pi],
               [-math.pi/4, 0], [-math.pi/2, -math.pi/4], [-math.pi*3/4, -math.pi/2], [-math.pi, -math.pi*3/4]]

    for q in quarter:
        a,b = q
        if start_radian <= end_radian:      #
            a_in_range = get_obs_check_radian(start_radian = start_radian, end_radian = end_radian, angle = a)
            b_in_range = get_obs_check_radian(start_radian = start_radian, end_radian = end_radian, angle = b)
            if not a_in_range and not b_in_range:
                if a <= start_radian <= end_radian <= b:
                    matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, start_radian, end_radian, value)
                else:
                    pass
            elif a_in_range and not b_in_range:
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, a, end_radian, value)
            elif not a_in_range and b_in_range:
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, start_radian, b, value)

            elif a_in_range and b_in_range:
                assert start_radian <= a <= b <= end_radian
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, a, b, value)



        elif start_radian > end_radian:
            start_radian1 = start_radian
            end_radian1 = math.pi
            start_radian2 = -math.pi
            end_radian2 = end_radian
            assert start_radian1 <= end_radian1 and start_radian2 <= end_radian2

            #range1
            a_in_range = get_obs_check_radian(start_radian = start_radian1, end_radian = end_radian1, angle = a)
            b_in_range = get_obs_check_radian(start_radian = start_radian1, end_radian = end_radian1, angle = b)
            if not a_in_range and not b_in_range:
                if a <= start_radian1 <= end_radian1 <= b:
                    matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, start_radian1, end_radian1, value)
                else:
                    pass
            elif a_in_range and not b_in_range:
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, a, end_radian1,value)
            elif not a_in_range and b_in_range:
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, start_radian1, b, value)
            elif a_in_range and b_in_range:
                assert start_radian1 <= a <= b <= end_radian1
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, a, b, value)

            #range2
            a_in_range = get_obs_check_radian(start_radian = start_radian2, end_radian = end_radian2, angle = a)
            b_in_range = get_obs_check_radian(start_radian = start_radian2, end_radian = end_radian2, angle = b)
            if not a_in_range and not b_in_range:
                if a <= start_radian2 <= end_radian2 <= b:
                    matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, start_radian2, end_radian2, value)
                else:
                    pass
            elif a_in_range and not b_in_range:
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, a, end_radian2, value)
            elif not a_in_range and b_in_range:
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, start_radian2, b, value)
            elif a_in_range and b_in_range:
                assert start_radian2 <= a <= b <= end_radian2
                matrix = quarter_circle_drawing(matrix, [cx, cy], arc_R/vis_clear, a, b, value)

    return matrix








def quarter_circle_drawing(matrix, circle_center, circle_r, start_radian, end_radian, value,rotate = True):
    """
    draw arc within angle range [0, pi/4], right is positive x, down is positive y
    start_radian and end_radian should be within the range of [0,4/pi], [4/pi, 2/pi], [2/pi,3pi/2],[3pi/2, pi]; [-4/pi, 0], [-pi/2, pi/4],
    [-3pi/2, -pi/2], [-pi, -3pi/2]
    if not rotate:

       |
       |--------
       |        |
    ------------|-------> x
       |        |
       |--------
       |
       v y

    elif rotate:

           ^ x
        ---|---
       |   |   |
       |   |   |
       |   |   |
    --------------> y
           |




    """
    if 0 <= start_radian < math.pi/4 and start_radian <= end_radian <= math.pi/4:
        quarter = '1'
    elif math.pi/4 <= start_radian < math.pi/2 and start_radian <= end_radian <= math.pi/2:
        quarter = '2'
    elif math.pi/2 <= start_radian < math.pi*3/4 and start_radian <= end_radian <= math.pi*3/4:
        quarter = '3'
    elif math.pi*3/4 <= start_radian <= math.pi and start_radian <= end_radian <= math.pi:
        quarter = '4'
    elif -math.pi <= start_radian < -math.pi*3/4 and start_radian <= end_radian <= -math.pi*3/4:
        quarter = '-4'
    elif -math.pi*3/4 <= start_radian < -math.pi/2 and start_radian <= end_radian <= -math.pi/2:
        quarter = '-3'
    elif -math.pi/2 <= start_radian < -math.pi/4 and start_radian <= end_radian <= -math.pi/4:
        quarter = '-2'
    elif -math.pi/4 <= start_radian < 0 and start_radian <= end_radian <= 0:
        quarter = '-1'
    else:
        raise NotImplementedError

    size = matrix.shape[0]
    #start from initial point
    x,y = circle_r, 0
    while x>=-y:
        left_p = [x-1, y-1]
        top_p = [x, y-1]
        d_left = (left_p[0])**2 + (left_p[1])**2 - circle_r**2
        d_top = (top_p[0])**2 + (top_p[1])**2 - circle_r**2
        P_k = d_left + d_top
        if P_k >= 0:
            next_x, next_y = x-1, y-1
        else:
            next_x, next_y = x, y-1

        if quarter == '1':
            if 0 < x + circle_center[0] < size and -size/2 < y+circle_center[1] < size/2:
                plot_y = y + size/2 + circle_center[1]
                plot_x = x + circle_center[0]

                current_radian = math.atan2(-y, x)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value


        elif quarter == '2':
            if 0 < -y + circle_center[0] < size and -size/2 < -x+circle_center[1] < size/2:        #2
                plot_y = -x + size/2 + circle_center[1]
                plot_x = -y  + circle_center[0]

                current_radian = math.atan2(x, -y)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value

        elif quarter == '3':
            if 0 < y + circle_center[0] < size and -size/2 < -x+circle_center[1] < size/2:        #3
                plot_y = -x + size/2 + circle_center[1]
                plot_x = y  + circle_center[0]

                current_radian = math.atan2(x, y)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value

        elif quarter == '4':
            if 0 < -x + circle_center[0] < size and -size/2 < y+circle_center[1] < size/2:   # 4
                plot_y = y + size/2 + circle_center[1]
                plot_x = -x + circle_center[0]

                current_radian = math.atan2(-y, -x)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value

        elif quarter == '-1':
            if 0 < x + circle_center[0] < size and -size/2 < -y+circle_center[1] < size/2:        #-1
                plot_y = -y + size/2 + circle_center[1]
                plot_x = x + circle_center[0]

                current_radian = math.atan2(y, x)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value

        elif quarter == '-2':
            if 0 < -y + circle_center[0] < size and -size/2 < x+circle_center[1] < size/2:        #-2
                plot_y = x + size/2 + circle_center[1]
                plot_x = -y  + circle_center[0]

                current_radian = math.atan2(-x, -y)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value

        elif quarter == '-3':
            if 0 < y + circle_center[0] < size and -size/2 < x+circle_center[1] < size/2:        #-3
                plot_y = x + size/2 + circle_center[1]
                plot_x = y  + circle_center[0]

                current_radian = math.atan2(-x, y)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value

        elif quarter == '-4':

            if 0 < -x + circle_center[0] < size and -size/2 < -y+circle_center[1] < size/2:   # -4

                plot_y = -y + size/2 + circle_center[1]
                plot_x = -x + circle_center[0]

                current_radian = math.atan2(y, -x)
                if  start_radian <= current_radian <= end_radian:
                    if rotate:
                        matrix[-int(plot_x)][int(plot_y)] = value
                    else:
                        matrix[int(plot_y)][int(plot_x)] = value




        x,y = next_x, next_y

    return matrix

plot_matrix = np.zeros((40,40))
draw_arc =  {'start point': [0.0, 50.25015644561822], 'end radian': 1.6208171836006666,
             'end point': [142.28756555322954, 100.0], 'start radian': 0.848062078981481,
             'arc start radian': 0.0, 'arc end radian': -1.7453292519943295, 'arc center': [10.0, 250.0], 'arc R': 200.0}


# draw_arc_dict = {'start radian': math.pi/2, 'end radian': -math.pi/2, "arc center": [90,-10], "arc R": 50}
# plot_matrix = bresenham_arc(plot_matrix, draw_arc, vis=200, vis_clear=5, value = 1)
## plot_matrix = quarter_circle_drawing(matrix=plot_matrix, circle_center = [0,10], circle_r = 20, start_radian=0., end_radian=math.pi/4, rotate = True)
## plot_matrix = quarter_circle_drawing(matrix=plot_matrix, circle_center = [0,10], circle_r = 20, start_radian=math.pi/4., end_radian=math.pi/2, rotate = True)
# plt.imshow(plot_matrix)
# plt.show()
#
# raise NotImplementedError



# def quarter_circle_drawing(matrix, start_point, start_radian, end_point, end_radian, circle_center, circle_r, vis_clear):
#     size = matrix.shape[0]
#     circle_center[0] /= vis_clear
#     circle_center[1] /= vis_clear
#     circle_r /= vis_clear
# 
#     x, y = (circle_center[0]+circle_r, circle_center[1])      #start from this
#     rad1 = 0
#     while x>=y:
#         left_p = (x-1, y+1)
#         top_p = (x, y+1)
#         d_left = (left_p[0]-circle_center[0])**2 + (left_p[1]-circle_center[1])**2 - circle_r**2
#         d_top = (top_p[0]-circle_center[0])**2 + (top_p[1]-circle_center[1])**2 - circle_r**2
#         P_k = d_left + d_top
#         if P_k >= 0:
#             x_next, y_next = x-1, y+1
#         else:
#             x_next, y_next = x, y+1
# 
#         if -size < x < size and 0<=y <size:
#             plot_x = x + size/2
#             plot_y = y
#             matrix[-int(plot_y)][int(plot_x)] = 1
# 
#         x, y = x_next, y_next
# 
#     return matrix
# 
# 
# plot_matrix = np.zeros((40,40))
# plot_matrix = quarter_circle_drawing(matrix = plot_matrix, start_point=[25, 5], start_radian=0, end_point = [], end_radian=0, circle_center=[-10, -10],
#                                      circle_r = 30, vis_clear=5)
# plt.imshow(plot_matrix)
# plt.show()
# raise NotImplementedError





class env_test(OlympicsBase):
    def __init__(self, map=gamemap):
        super(env_test, self).__init__(map)

        self.gamma = 1  # v衰减系数
        self.restitution = 1
        self.print_log = False
        self.tau = 0.1
        self.draw_obs = True
        self.show_traj = False

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

        actions_list = self.check_action(actions_list)

        self.stepPhysics(actions_list, self.step_cnt)
        self.step_cnt += 1
        step_reward = 1 #self.get_reward()
        obs_next = self.get_obs()
        # obs_next = 1
        done = False #self.is_terminal()
        #check overlapping
        #self.check_overlap()
        # self.get_obs2()

        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return obs_next, step_reward, done, ''

    def get_obs(self):
        self.obs_boundary = list()

        obs_list = list()

        for agent_idx, agent in enumerate(self.agent_list):
            if self.agent_list[agent_idx].type == 'ball':
                self.obs_boundary.append(None)
                obs_list.append(None)
                continue

            time_s = time.time()
            theta_copy = self.agent_theta[agent_idx][0]
            agent_pos = self.agent_pos
            agent_x, agent_y = agent_pos[agent_idx][0], agent_pos[agent_idx][1]
            theta = theta_copy
            if theta != 0:
                theta = abs(theta)%360 * (theta/abs(theta))     #normalise the angle
            position_init = agent.position_init

            visibility = self.agent_list[agent_idx].visibility
            v_clear = self.agent_list[agent_idx].visibility_clear
            # obs_map = np.zeros((visibility[0], visibility[1]))
            # obs_weight,obs_height = int(visibility[0]/v_clear[0]),int(visibility[1]/v_clear[1])
            obs_size = int(visibility / v_clear)

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
                x_new_ = x_new - vec_oo_[0]
                y_new_ = y_new - vec_oo_[1]
                agent_current_boundary.append([x_new_, -y_new_])
            self.obs_boundary.append(agent_current_boundary)

            #compute center of view
            view_center_x = agent_x + visibility/2*math.cos(theta*math.pi/180)      #start from agent x,y
            view_center_y = agent_y + visibility/2*math.sin(theta*math.pi/180)
            view_center = [view_center_x, view_center_y]
            view_R = visibility*math.sqrt(2)/2
            #compute closest distance from view center to the line
            object_consider = []
            for c in self.map['objects']:
                if (c.type == "wall") or (c.type == "cross"):
                    closest_dist = distance_to_line(c.init_pos[0], c.init_pos[1], view_center)
                    if closest_dist <= view_R:
                        object_consider.append(c)

                elif c.type == 'arc':
                    arc_center = c.center
                    center_dist = (arc_center[0]-view_center[0])**2 + (arc_center[1]-view_center[1])**2
                    if (c.R - view_R)**2 <= center_dist <= (c.R + view_R)**2:
                        object_consider.append(c)
                        # pass
                else:
                    raise NotImplementedError

            obs_map = np.zeros((obs_size,obs_size))
            #rotating the object
            for obj in object_consider:
                if obj.type == 'wall' or obj.type == 'cross':
                    current_pos = obj.init_pos
                    obj.rotate_pos = []
                    for end_point in current_pos:
                        # px = end_point[0]-agent_x
                        # py = end_point[1]-agent_y
                        #
                        # nx = [math.cos(theta*math.pi/180), math.sin(theta*math.pi/180)]
                        # ny = [-math.sin(theta*math.pi/180), math.cos(theta*math.pi/180)]
                        # new_x = px*nx[0] + py*nx[1]
                        # new_y = px*ny[0] + py*ny[1]
                        # obj.rotate_pos.append([new_x, new_y])
                        obj.rotate_pos.append(point_rotate([agent_x, agent_y], end_point, theta))
                        # obj.rotate_pos.append(coordinate_rotate([agent_x, agent_y], -theta, end_point))

                    # compute the intersection point
                    intersect_p = []
                    rotate_boundary = [[[0, -visibility / 2], [0, visibility / 2]],
                                       [[0, visibility / 2], [visibility, visibility / 2]],
                                       [[visibility, visibility / 2], [visibility, -visibility / 2]],
                                       [[visibility, -visibility / 2], [0, -visibility / 2]]]

                    # obs_rotate_boundary = []              #debug rotate boundard
                    # for line in self.obs_boundary:
                    #     rotate_bound = [point_rotate([agent_x, agent_y], i, theta) for i in line]
                    #     obs_rotate_boundary.append(rotate_bound)

                    # for line in self.obs_boundary:
                    for line in rotate_boundary:
                        _intersect_p = line_intersect(line1=line, line2=obj.rotate_pos, return_p=True)
                        if _intersect_p:
                            # intersect_p.append({"bound": line, "intersect point": intersect_p})
                            intersect_p.append(_intersect_p)

                    draw_line = []
                    if len(intersect_p) == 0:
                        # no intersection, maybe true or maybe the whole line segment is within the view
                        point_1_in_view=  0 < obj.rotate_pos[0][0] < visibility and abs(obj.rotate_pos[0][1]) < visibility / 2
                        point_2_in_view = 0 < obj.rotate_pos[1][0] < visibility and abs(obj.rotate_pos[1][1]) < visibility / 2

                        if point_1_in_view and point_2_in_view:
                            draw_line.append(obj.rotate_pos[0])
                            draw_line.append(obj.rotate_pos[1])
                        elif not point_1_in_view and not point_2_in_view:
                            continue
                        else:
                            raise NotImplementedError

                        # continue
                    elif len(intersect_p) == 1:
                        # one intersectoin, rotate it first
                        # intersect_p1 = intersect_p[0]
                        # c_p1 = [intersect_p1[0]-agent_x, intersect_p1[1]-agent_y]
                        # rotate_intersect_p1 = [c_p1[0]*nx[0]+c_p1[1]*nx[1], c_p1[0]*ny[0]+c_p1[1]*ny[1]]
                        # draw_line.append(rotate_intersect_p1)

                        draw_line.append(intersect_p[0])

                        if 0 < obj.rotate_pos[0][0] < visibility and abs(obj.rotate_pos[0][1]) < visibility / 2:
                            draw_line.append(obj.rotate_pos[0])
                        elif 0 < obj.rotate_pos[1][0] < visibility and abs(obj.rotate_pos[1][1]) < visibility / 2:
                            draw_line.append(obj.rotate_pos[1])
                        else:
                            # only one point in the view
                            pass
                            # raise NotImplementedError

                    elif len(intersect_p) == 2:

                        draw_line.append(intersect_p[0])
                        draw_line.append(intersect_p[1])

                    # start drawing the object
                    obs_map = DDA_line(obs_map, draw_line, visibility, v_clear, value=COLOR_TO_IDX[obj.color])


                elif obj.type == 'arc':
                    # continue
                    current_center = obj.center
                    current_R = obj.R
                    current_start_radian, current_end_radian = obj.start_radian, obj.end_radian
                    obj.rotate_center = point_rotate([agent_x, agent_y], current_center, theta)
                    rotate_start_radian = current_start_radian + theta/180*math.pi
                    rotate_end_radian = current_end_radian + theta/180*math.pi

                    if rotate_start_radian > math.pi:
                        rotate_start_radian -= 2*math.pi
                    elif rotate_start_radian < -math.pi:
                        rotate_start_radian += 2*math.pi

                    if rotate_end_radian > math.pi:
                        rotate_end_radian -= 2*math.pi
                    elif rotate_end_radian < -math.pi:
                        rotate_end_radian += 2*math.pi

                    assert -math.pi <= rotate_start_radian <= math.pi, print(rotate_start_radian)
                    assert -math.pi <= rotate_end_radian <= math.pi, print(rotate_end_radian)

                    obj.rotate_start_radian = rotate_start_radian
                    obj.rotate_end_radian = rotate_end_radian

                    intersect_p = []
                    rotate_boundary = [[[0, -visibility / 2], [0, visibility / 2]],
                                       [[0, visibility / 2], [visibility, visibility / 2]],
                                       [[visibility, visibility / 2], [visibility, -visibility / 2]],
                                       [[visibility, -visibility / 2], [0, -visibility / 2]]]

                    for line in rotate_boundary:
                        arc_intersect = arc_line_intersect(line=line, arc_center=obj.rotate_center, arc_start_radian=obj.rotate_start_radian,
                                                           arc_end_radian=obj.rotate_end_radian, arc_R = current_R)
                        intersect_p = intersect_p + arc_intersect

                    # assert len(intersect_p) <= 2, print(intersect_p)

                    draw_arc = []
                    if len(intersect_p) == 0:
                        continue
                    elif len(intersect_p) == 1:
                        draw_arc.append(intersect_p[0])
                        #add the next end
                        start_radian_rotate_pos = [obj.rotate_center[0]+obj.R*math.cos(obj.rotate_start_radian),
                                                   obj.rotate_center[1]+obj.R*math.sin(obj.rotate_start_radian)]
                        end_radian_rotate_pos = [obj.rotate_center[0]+obj.R*math.cos(obj.rotate_end_radian),
                                                   obj.rotate_center[1]+obj.R*math.sin(obj.rotate_end_radian)]

                        if 0 < start_radian_rotate_pos[0] < visibility and abs(start_radian_rotate_pos[1]) < visibility / 2:
                            draw_arc_dict = {'start point': start_radian_rotate_pos,
                                             'start radian': obj.rotate_start_radian,
                                             'end point': intersect_p[0]['point'], 'end radian':intersect_p[0]['rad'],
                                             "arc start radian": obj.rotate_start_radian, "arc end radian": obj.rotate_end_radian,
                                             "arc center": obj.rotate_center, 'arc R': obj.R}

                        elif 0 < end_radian_rotate_pos[0] < visibility and abs(end_radian_rotate_pos[1]) < visibility / 2:
                            draw_arc_dict = {'start point': intersect_p[0]['point'],
                                             'start radian': intersect_p[0]['rad'],
                                             'end point': end_radian_rotate_pos, 'end radian':obj.rotate_end_radian,
                                             "arc start radian": obj.rotate_start_radian, "arc end radian": obj.rotate_end_radian,
                                             "arc center": obj.rotate_center, 'arc R': obj.R}
                        else:
                            draw_arc_dict = {'start point': intersect_p[0]['point'],
                                             'start radian': intersect_p[0]['rad'],
                                             "arc start radian": obj.rotate_start_radian, "arc end radian": obj.rotate_end_radian,
                                             "arc center": obj.rotate_center, 'arc R': obj.R}

                    elif len(intersect_p) == 2:
                        max_rad = max(intersect_p[0]['rad'], intersect_p[1]['rad'])
                        min_rad = min(intersect_p[0]['rad'], intersect_p[1]['rad'])
                        if obj.rotate_start_radian <= obj.rotate_end_radian:
                            start_rad = min_rad
                            end_rad = max_rad
                        else:
                            if max_rad > 0 and min_rad <0:
                                start_rad = max_rad
                                end_rad = min_rad
                            else:
                                start_rad = min_rad
                                end_rad = max_rad
                        draw_arc_dict = {'start radian': start_rad,
                                         'end radian': end_rad,
                                         'arc start radian': obj.rotate_start_radian, 'arc end radian': obj.rotate_end_radian,
                                         "arc center": obj.rotate_center, 'arc R': obj.R}


                        # draw_arc_dict = {'start point': intersect_p[0]['point'],
                        #                  'start radian': intersect_p[0]['rad'],
                        #                  'end point': intersect_p[1]['point'],
                        #                  'end radian': intersect_p[1]['rad'],
                        #                  'arc start radian': obj.rotate_start_radian, 'arc end radian': obj.rotate_end_radian,
                        #                  "arc center": obj.rotate_center, 'arc R': obj.R}

                    else:
                        #if there exists multiple intersection, we find the two extreme point radian

                        if obj.rotate_start_radian > 0 and obj.rotate_end_radian < 0:
                            raise NotImplementedError

                        elif obj.rotate_start_radian < 0 and obj.rotate_end_radian < 0 and obj.rotate_start_radian > obj.rotate_end_radian:
                            raise NotImplementedError
                        elif obj.rotate_start_radian > 0 and obj.rotate_end_radian > 0 and obj.rotate_start_radian > obj.rotate_end_radian:
                            raise NotImplementedError

                        else:
                            start_p = None
                            end_p = None

                            for p in intersect_p:
                                if start_p is None and end_p is None:
                                    start_p = p
                                    end_p = p
                                    continue


                                temp_radian = p['rad']
                                if abs(temp_radian-obj.rotate_start_radian) < abs(start_p['rad'] - obj.rotate_start_radian):
                                    start_p = p
                                if abs(temp_radian-obj.rotate_end_radian) < abs(end_p['rad'] - obj.rotate_end_radian):
                                    end_p = p

                            draw_arc_dict = {'start point': start_p['point'], 'start radian':start_p['rad'],
                                             'end point': end_p['point'], 'end radian': end_p['rad'],
                                             'arc start radian': obj.rotate_start_radian, 'arc end radian': obj.rotate_end_radian,
                                             'arc center': obj.rotate_center, 'arc R': obj.R}



                    print('draw arc = ', draw_arc_dict)
                    obs_map = bresenham_arc(obs_map, draw_arc_dict, visibility, v_clear, value=COLOR_TO_IDX[obj.color])


                    # elif len(intersect_p) == 1:
                    #     #find other end of the arc that is in the view
                    #     draw_arc.append(intersect_p[0])




                else:
                    raise NotImplementedError


            obs_list.append(obs_map)

        self.obs_list = obs_list



    def get_obs1(self):
        """
        POMDP: partial observation
        step1: 将原坐标系上landmark（Cross, Wall...）映射到新坐标系，生成新坐标
        step2: 在新坐标系下，进行栅格化
        step3: 在视野范围内，填值
        """



        # this is for debug
        self.obs_boundary = list()


        obs_list = list()

        for agent_idx, agent in enumerate(self.agent_list):
            if self.agent_list[agent_idx].type == 'ball':
                self.obs_boundary.append(None)
                obs_list.append(None)
                continue

            time_s = time.time()
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

            #compute center of view
            view_center_x = agent_x + visibility/2*math.cos(theta*math.pi/180)      #start from agent x,y
            view_center_y = agent_y + visibility/2*math.sin(theta*math.pi/180)
            view_center = [view_center_x, view_center_y]
            view_R = visibility*math.sqrt(2)/2
            line_consider = []

            # 计算GameMap上objects组件，相对于agent的坐标(没有旋转)
            for index_m, item in enumerate(self.map["objects"]):
                if (item.type == "wall") or (item.type == "cross"):
                    closest_dist = distance_to_line(item.init_pos[0], item.init_pos[1], view_center)
                    if closest_dist <= view_R:
                        line_consider.append(item)

                    # pos = item.init_pos
                    # item.cur_pos = list()
                    # for index, p in enumerate(pos):
                    #     vec_o_d = (p[0], -p[1])
                    #     vec_oo_ = (-agent_x, agent_y)
                    #     vec_od = (vec_o_d[0] + vec_oo_[0], vec_o_d[1] + vec_oo_[1])
                    #     item.cur_pos.append([vec_od[0], vec_od[1]])
                elif item.type == "arc":
                    pos = item.center
                    item.cur_pos = list()
                    vec_o_d = (pos[0], -pos[1])
                    vec_oo_ = (-agent_x, agent_y)
                    vec_od = (vec_o_d[0] + vec_oo_[0], vec_o_d[1] + vec_oo_[1])
                    item.cur_pos.append([vec_od[0], vec_od[1]])

            # 计算视野中心点到各个边的距离
            # 视野中心点
            if self.VIEW_ITSELF:
                vec_oc = (visibility/2, 0)
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
                    # distance = abs(get_distance(c.cur_pos, vec_oc_, c.length, pixel=False))
                    # if distance <= visibility/2 * 1.415:
                    # # if distance <= 50 * 1.415:
                    #     map_deduced["objects"].append(c)
                    #     map_objects.append(c)
                elif c.type == "arc":
                    distance = distance_2points([c.cur_pos[0][0]-vec_oc_[0],c.cur_pos[0][1]-vec_oc_[1]])
                    if distance <= visibility/2 * 1.415 + c.R:
                        map_deduced["objects"].append(c)
                        map_objects.append(c)
                else:
                    raise ValueError("No such object type- {}. Please check scenario.json".
                                     format(c.type))

            map_deduced["agents"] = list()

            # 当前agent自己
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

            # obs_map = np.zeros(( obs_weight , obs_height))
            obs_map = np.zeros((obs_size,obs_size))
            for obj in map_deduced["objects"]:
                if (obj.type == "wall") or (obj.type == "cross"):
                    raise NotImplementedError
                    # points_pos = obj.cur_pos
                    # obj.cur_pos_rotated = list()
                    # for pos in points_pos:
                    #     pos_x = pos[0]
                    #     pos_y = pos[1]
                    #     theta_obj = - theta
                    #     pos_x_, pos_y_ = rotate2(pos_x, pos_y, theta_obj)
                    #     obj.cur_pos_rotated.append([pos_x_, pos_y_])
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

            time_stamp = time.time()
            #for component in map_objects:
            for component in list(reversed(map_objects)):           #reverse to consider agent first, then wall
                for i in range(obs_size):
                    if self.VIEW_ITSELF:
                        x = visibility - v_clear*i - v_clear/2
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


                        if obs_map[i][j] > 0:           #when there is already object on this pixel
                            continue
                        else:
                            if (component.type == "wall") or (component.type == "cross"):
                                raise NotImplementedError
                                # distance = abs(get_distance(component.cur_pos_rotated, point, component.length,
                                #                             pixel=True))
                                # if distance <= v_clear :  # 距离小于等于1个像素点长度
                                #     obs_map[i][j] = COLOR_TO_IDX[component.color]
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
                                    if (distance <= radius + v_clear) and (distance >= radius - v_clear):
                                        obs_map[i][j] = COLOR_TO_IDX[component.color]


            #now start drawing line
            for obj in line_consider:
                if obj.type == 'wall' or obj.type == 'cross':
                    current_pos = obj.init_pos
                    obj.rotate_pos = []
                    for end_point in current_pos:
                        # px = end_point[0]-agent_x
                        # py = end_point[1]-agent_y
                        #
                        # nx = [math.cos(theta*math.pi/180), math.sin(theta*math.pi/180)]
                        # ny = [-math.sin(theta*math.pi/180), math.cos(theta*math.pi/180)]
                        # new_x = px*nx[0] + py*nx[1]
                        # new_y = px*ny[0] + py*ny[1]
                        # obj.rotate_pos.append([new_x, new_y])
                        obj.rotate_pos.append(point_rotate([agent_x, agent_y], end_point, theta))
                        # obj.rotate_pos.append(coordinate_rotate([agent_x, agent_y], -theta, end_point))

                    # compute the intersection point
                    intersect_p = []
                    rotate_boundary = [[[0, -visibility / 2], [0, visibility / 2]],
                                       [[0, visibility / 2], [visibility, visibility / 2]],
                                       [[visibility, visibility / 2], [visibility, -visibility / 2]],
                                       [[visibility, -visibility / 2], [0, -visibility / 2]]]

                    # obs_rotate_boundary = []              #debug rotate boundard
                    # for line in self.obs_boundary:
                    #     rotate_bound = [point_rotate([agent_x, agent_y], i, theta) for i in line]
                    #     obs_rotate_boundary.append(rotate_bound)

                    # for line in self.obs_boundary:
                    for line in rotate_boundary:
                        _intersect_p = line_intersect(line1=line, line2=obj.rotate_pos, return_p=True)
                        if _intersect_p:
                            # intersect_p.append({"bound": line, "intersect point": intersect_p})
                            intersect_p.append(_intersect_p)

                    draw_line = []
                    if len(intersect_p) == 0:
                        # no intersection
                        continue
                    elif len(intersect_p) == 1:
                        # one intersectoin, rotate it first
                        # intersect_p1 = intersect_p[0]
                        # c_p1 = [intersect_p1[0]-agent_x, intersect_p1[1]-agent_y]
                        # rotate_intersect_p1 = [c_p1[0]*nx[0]+c_p1[1]*nx[1], c_p1[0]*ny[0]+c_p1[1]*ny[1]]
                        # draw_line.append(rotate_intersect_p1)

                        draw_line.append(intersect_p[0])

                        if 0 < obj.rotate_pos[0][0] < visibility and abs(obj.rotate_pos[0][1]) < visibility / 2:
                            draw_line.append(obj.rotate_pos[0])
                        elif 0 < obj.rotate_pos[1][0] < visibility and abs(obj.rotate_pos[1][1]) < visibility / 2:
                            draw_line.append(obj.rotate_pos[1])
                        else:
                            # only one point in the view
                            pass
                            # raise NotImplementedError

                    elif len(intersect_p) == 2:

                        draw_line.append(intersect_p[0])
                        draw_line.append(intersect_p[1])

                    obs_map = DDA_line(obs_map, draw_line, visibility, v_clear, value=COLOR_TO_IDX[obj.color])

                else:
                    raise NotImplementedError


            obs_list.append(obs_map)
            if self.print_log2:
                print('agent {} get obs duration {}'.format(agent_idx, time.time() - time_stamp))
        self.obs_list = obs_list


        return obs_list




import random

env = env_test()


for _ in range(100):

    env.reset()
    env.render()
    # env.agent_theta[0][0] = 180
    done = False
    step = 0
    while not done:
        # print('\n step = ', step)
        #if step < 10:
        #    action = [[random.randint(-100,200),random.randint(-30, 30)]]#, [2,1]]#, [2,2]]#, [2,1]]#[[2,1], [2,1]] + [ None for _ in range(4)]
        #else:
        #    action = [[random.randint(-100,200),random.randint(-30, 30)]]#, [2,1]]#, [2,1]]#, [2,random.randint(0,2)]] #[[2,1], [2,1]] + [None for _ in range(4)]
        #action1 = [random.randint(-100, 200), random.randint(-30, 30)]
        #action2 = [random.randint(-100, 200), random.randint(-30, 30)]
        # action = [[random.uniform(0,1)*10, random.uniform(-30,30)]]
        # action2 = [100+random.uniform(0,1)*15, 0]
        # action = [[200, 0] for _ in range(1)]
        action = [[0,0]]
        # action = [action]
        _,_,done, _ = env.step(action)

        # print('agent v = ', env.agent_v)
        env.render()
        step += 1
        time.sleep(0.05)
        # if step < 60:
        #     time.sleep(0.05)
        # else:
        #     time.sleep(0.05)


