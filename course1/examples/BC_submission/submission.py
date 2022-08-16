from pathlib import Path
import os

current_path = Path(__file__).resolve().parent
model_path = os.path.join(current_path, 'actor_state_dict.pt')



import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1600, 400),
            nn.ReLU(),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, X):
        action_batch = self.net(X)
        action_batch[:, 0] = torch.tanh(action_batch[:,0])*150+50
        action_batch[:, 1] = torch.tanh(action_batch[:, 1])*30
        return action_batch

model = Net()
loaded_actor_state = torch.load(model_path)
model.load_state_dict(loaded_actor_state)

def my_controller(observation, action_space, is_act_continuous=True):

    obs_array = torch.tensor(observation['obs']['agent_obs']).float().reshape(1, -1)
    action = model(obs_array)

    return [[action[0][0].item()], [action[0][1].item()]]

