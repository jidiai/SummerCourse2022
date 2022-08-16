# -*- coding:utf-8  -*-
# Author: Shu LIN

import numpy

DEPTH = 4  # 搜索深度
INF = 100000000
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量


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
    return score


# 使用Alpha-Beta剪枝搜索，返回最优结果和最优动作
def alphabeta(board, depth, alpha, beta, color, width, height):
    if depth == 0:
        return evaluate(board, color), -1, -1
    x = y = -1
    noMove = True
    for i in range(width):
        for j in range(height):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color, width, height):
                    noMove = False
                    v = -alphabeta(newBoard, depth - 1, -beta, -alpha, -color, width, height)[0]
                    if v > alpha:
                        if beta <= alpha:
                            return v, i, j
                        alpha = v
                        x, y = i, j
    if noMove:
        v = -alphabeta(board, depth - 1, -beta, -alpha, -color, width, height)[0]
        if v > alpha:
            alpha = v
    return alpha, x, y


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
    _, x, y = alphabeta(board, DEPTH, -INF, INF, myColor, width, height)
    return wrap_action(x, y, height, width)
