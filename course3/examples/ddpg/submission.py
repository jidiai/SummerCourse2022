# -*- coding:utf-8  -*-
# Time  : 2022/8/19 下午5:24
# Author: Yahui Cui

import random
import os
import numpy as np
import collections
import torch
from torch import nn
import torch.nn.functional as F


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(torch.nn.Module):
    # 这是一个简单的两层神经网络
    def __init__(self,
                 num_in,
                 num_out,
                 hidden_dim,
                 activation=F.relu,
                 out_fn=lambda x: x):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)

        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim,
                 discrete, action_bound, sigma, actor_lr, critic_lr, tau,
                 gamma, device):
        # self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        out_fn = (lambda x: x) if discrete else (
            lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(num_in_actor,
                                num_out_actor,
                                hidden_dim,
                                activation=F.relu,
                                out_fn=out_fn).to(device)
        self.target_actor = TwoLayerFC(num_in_actor,
                                       num_out_actor,
                                       hidden_dim,
                                       activation=F.relu,
                                       out_fn=out_fn).to(device)
        self.critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 1,
                                        hidden_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = num_out_actor
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        if not SUBMISSION:
            action = action + self.sigma * np.random.randn(self.action_dim)
        else:
            action = np.array([action])
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(
            torch.cat(
                [next_states, self.target_actor(next_states)], dim=1))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(
            F.mse_loss(
                # MSE损失函数
                self.critic(torch.cat([states, actions], dim=1)),
                q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(
            self.critic(
                # 策略网络就是为了使得Q值最大化
                torch.cat([states, self.actor(states)], dim=1)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def save(self, save_path, episode):
        base_path = save_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(int(episode)) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

    def load(self, file):
        self.actor.load_state_dict(torch.load(file, map_location='cpu'))


SUBMISSION = False
actor_lr = 5e-4
critic_lr = 5e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


file = os.path.join(os.path.dirname(__file__), 'actor_200.pth')
replay_buffer = ReplayBuffer(buffer_size)
state_dim = 3
action_dim = 1
action_bound = 2
agent = DDPG(state_dim, action_dim, state_dim + action_dim, hidden_dim, False,
             action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
if SUBMISSION:
    agent.load(file)


def my_controller(observation, action_space, is_act_continuous=True):
    action = agent.take_action(observation['obs'])
    action2env = [action]
    return action2env

