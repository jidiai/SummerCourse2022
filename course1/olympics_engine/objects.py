import math

# # Used to map colors to integers
# COLOR_TO_IDX = {
#     'red': 7,
#     'green': 1,
#     'sky blue': 2,
#     'orange': 3,
#     'grey': 4,
#     'purple': 5,
#     'black': 6,
#     'light green': 0,
#     'blue': 8
# }
#
# # Map of object type to integers
# OBJECT_TO_IDX = {
#     'agent': 0,
#     'wall': 1,  # 反弹
#     'cross': 2,   # 可穿越
#     'goal': 3,   # 可穿越 # maybe case by case
#     'arc': 4,
#     'ball': 5
# }

from olympics_engine.tools.func import closest_point
from olympics_engine.tools.settings import *



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


class Ball(GameObj):
    def __init__(self, mass = 1, r = 20, position = None, color = 'purple',
                 vis = None, vis_clear = None):
        super(Ball, self).__init__(type = 'ball', color = color)
        self.position_init = position
        self.mass = mass
        self.r = r

        self.alive = True
        self.finished = False
        self.original_color = color

        self.energy = None
        self.visibility = None
        self.visibility_clear = None

    def reset(self):
        self.alive = True
        self.finished = False

    def reset_color(self):
        self.color = self.original_color

    def can_bounce(self):
        return True

    def render(self):
        pass


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





