# -*- coding:utf-8  -*-
# Author: Shu LIN

import numpy
import random

TIMES = 10000  # 模拟次数
INF = 100000000
EPS = 0.1
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量

visits = {}
returns = {}


# 放置棋子，计算新局面
def place(board, x, y, color, width, height):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < width and 0 <= j and j < height and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < width and 0 <= j and j < height and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid


# 评估局面
def evaluate(board, color, width, height):
    score = 0
    for i in range(width):
        for j in range(height):
            score += board[i][j] * color
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


# 选择下一步行动
def getMove(board, color, chooseBest, width, height):
    moves = []
    for i in range(width):
        for j in range(height):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color, width, height):
                    moves.append((i, j))
    if len(moves) == 0:
        return -1, -1
    best = -INF
    x = y = -1
    for (i, j) in moves:
        avg = INF
        if (color, i, j) in visits:
            avg = returns[color, i, j] / visits[color, i, j]
        if avg > best:
            best = avg
            x, y = i, j
    if chooseBest or random.random() > EPS:
        return x, y
    return random.choice(moves)


# 蒙特卡洛模拟
def simulate(board, color, width, height):
    x, y = getMove(board, color, False, width, height)
    noMove = x < 0
    if noMove:
        color = -color
        x, y = getMove(board, color, False, width, height)
        if x < 0:
            return evaluate(board, -color, width, height)
    newBoard = board.copy()
    place(newBoard, x, y, color, width, height)
    result = -simulate(newBoard, -color, width, height)
    global visits, returns
    if (color, x, y) not in visits:
        visits[color, x, y] = 1
        returns[color, x, y] = result
    else:
        visits[color, x, y] += 1
        returns[color, x, y] += result
    if noMove:
        return -result
    return result


# 使用蒙特卡洛树搜索，返回最优结果和最优动作
def montecarlo(board, color, width, height):
    for _ in range(TIMES):
        simulate(board, color, width, height)
    return getMove(board, color, True, width, height)


def wrap_action(x, y, width, height):
    action = [[0] * width, [0] * height]
    action[0][x] = 1
    action[1][y] = 1
    return action


def my_controller(observation, action_space, is_act_continuous=False):
    myColor = 1 if observation["chess_player_idx"] == 1 else -1
    height = observation["board_height"]
    width = observation["board_width"]
    board = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(width):
        for j in range(height):
            board[i][j] = 0
    for position in observation[1]:
        board[position[0]][position[1]] = 1
    for position in observation[2]:
        board[position[0]][position[1]] = -1
    x, y = montecarlo(board, myColor, height, width)
    return wrap_action(x, y, height, width)
