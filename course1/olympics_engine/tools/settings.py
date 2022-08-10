
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




