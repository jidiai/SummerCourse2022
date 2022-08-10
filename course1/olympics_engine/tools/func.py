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
