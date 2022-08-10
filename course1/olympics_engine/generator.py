import json
import sys
import os
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(father_path))

module = __import__("objects")

def create_scenario(scenario_name, file_path = None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'scenario.json')

    with open(file_path) as f:
        conf = json.load(f)[scenario_name]

    GameMap = dict()
    GameMap["objects"] = list()
    GameMap["agents"] = list()
    GameMap["view"] = conf["view"]

    for type in conf:
        if type == 'env_cfg':
            env_cfg_dict = conf[type]
            GameMap["env_cfg"] = env_cfg_dict
        elif type == 'obs_cfg':
            obs_cfg_dict = conf[type]
            GameMap["obs_cfg"] = obs_cfg_dict

        elif (type == "wall") or (type == "cross"):
            #print("!!", conf[type]["objects"])
            for key, value in conf[type]["objects"].items():
                GameMap["objects"].append(getattr(module, type.capitalize())
                     (
                     init_pos=value["initial_position"],
                    length=None,
                     color=value["color"],
                     ball_can_pass = value['ball_pass'] if ("ball_pass" in value.keys()
                                                            and value['ball_pass']=="True") else False,
                    width=value['width'] if ('width' in value.keys()) else None
                 )
                 )
        elif type == 'arc':
            for key, value in conf[type]['objects'].items():
                #print("passable = ", bool(value['passable']))
                GameMap['objects'].append(getattr(module, type.capitalize())(
                    init_pos = value["initial_position"],
                    start_radian = value["start_radian"],
                    end_radian = value["end_radian"],
                    passable = True if value["passable"] == "True" else False,
                    color = value['color'],
                    collision_mode=value['collision_mode'],
                    width = value['width'] if ("width" in value.keys()) else None
                ))

        elif type in ["agent","ball"]:
            for key, value in conf[type]["objects"].items():
                GameMap["agents"].append(getattr(module, type.capitalize())
                     (
                     mass=value["mass"],
                     r=value["radius"],
                     position=value["initial_position"],
                    color=value["color"],
                    vis = value["vis"] if ("vis" in value.keys()) else None,
                    vis_clear = value["vis_clear"] if ("vis_clear" in value.keys()) else None
                 ),
                                           )
    # print(" ========================== check GameMap ==========================")
    #print(GameMap)
    return GameMap