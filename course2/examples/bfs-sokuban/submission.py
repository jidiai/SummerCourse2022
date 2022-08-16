# -*- coding:utf-8  -*-
# Author: Shu LIN

import queue

DX = [-1, 1, 0, 0]
DY = [0, 0, -1, 1]
board = None


def encode(player, boxes, height, width):
    code = player[0] * width + player[1]
    bcodes = []
    for box in boxes:
        bcodes.append(box[0] * width + box[1])
    list.sort(bcodes)
    for bcode in bcodes:
        code = code * height * width + bcode
    return code


def decode(code, height, width, n):
    boxes = []
    for _ in range(n):
        boxes.append([code // width % height, code % width])
        code //= height * width
    player = [code // width, code % width]
    return player, boxes


def is_valid(x, y, height, width, board, check_box=False):
    return 0 <= x <= height and 0 <= y <= width and board[x][y] != 1 and (not check_box or board[x][y] != 0)


def get_box(x, y, boxes):
    for box in boxes:
        if box[0] == x and box[1] == y:
            return box
    return None


def is_terminal(boxes, board):
    for box in boxes:
        if board[box[0]][box[1]] != 2:
            return False
    return True


def step(player, boxes, action, label, height, width, board, visited, reached):
    x = player[0] + DX[action]
    y = player[1] + DY[action]
    if is_valid(x, y, height, width, board):
        box = get_box(x, y, boxes)
        if box is not None:
            bx = box[0] + DX[action]
            by = box[1] + DY[action]
            if not is_valid(bx, by, height, width, board, True) or get_box(bx, by, boxes) is not None:
                return False
            box[0] = bx
            box[1] = by
        state = encode([x, y], boxes, height, width)
        if state not in visited:
            if is_terminal(boxes, board):
                return True
            visited.add(state)
            reached.put([state, label])
        if box is not None:
            box[0] -= DX[action]
            box[1] -= DY[action]
    return False


def wrap_action(label):
    action = [0] * 4
    action[label] = 1
    return [action]


def my_controller(observation, action_space, is_act_continuous=False):
    height = observation["board_height"]
    width = observation["board_width"]
    state_map = observation["state_map"]

    # 广度优先搜索，将箱子禁止到达的位置标记为board[i][j] = -1
    reached = queue.Queue()
    global board
    if board is None:
        board = [[0 for _ in range(width)] for _ in range(height)]
        for i in range(height):
            for j in range(width):
                board[i][j] = state_map[i][j][0]
        for i in range(height):
            for j in range(width):
                if board[i][j] == 2:
                    reached.put([i, j])
                elif board[i][j] >= 3:
                    board[i][j] = 0
        while not reached.empty():
            x, y = reached.get()
            for action in range(4):
                bx = x + DX[action]
                by = y + DY[action]
                if is_valid(bx, by, height, width, board) and board[bx][by] == 0:
                    px = bx + DX[action]
                    py = by + DY[action]
                    if is_valid(px, py, height, width, board):
                        board[bx][by] = -1
                        reached.put([bx, by])
    boxes = []
    for i in range(height):
        for j in range(width):
            if state_map[i][j][0] == 3:
                boxes.append([i, j])
            elif state_map[i][j][0] == 4:
                player = [i, j]

    # 特殊判断单步结束的情况
    visited = {encode(player, boxes, height, width)}
    for action in range(4):
        if step(player, boxes, action, action, height, width, board, visited, reached):
            return wrap_action(action)
    n = len(boxes)

    # 广度优先搜索，找到这一步的最优动作
    while True:
        state, label = reached.get()
        player, boxes = decode(state, height, width, n)
        for action in range(4):
            if step(player, boxes, action, label, height, width, board, visited, reached):
                return wrap_action(label)
