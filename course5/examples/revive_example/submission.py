# -*- coding:utf-8  -*-
# Time  : 2022/7/26 下午4:24
# Author: Yahui Cui
import pickle
import numpy as np
from pathlib import Path
import os


class VenPolicy():
    """Strategies for using environment initialization"""
    def __init__(self, policy_model_path):
        self.policy_model = pickle.load(open(policy_model_path, 'rb'))

    def act(self, state):
        new_state = {}
        new_state['temperature'] = np.array([state])
        new_state['door_open'] = np.array([0])

        try:
            next_state = self.policy_model.infer(new_state)
        except:
            next_state = self.policy_model.infer_one_step(new_state)["action"]

        return next_state[0]


dirname = str(Path(__file__).resolve().parent)
model_path = os.path.join(dirname, "revive_policy.pkl")
agent = VenPolicy(model_path)


def my_controller(observation, action_space, model=None, is_act_continuous=False):
    action = agent.act(observation['obs'])
    return [action]

