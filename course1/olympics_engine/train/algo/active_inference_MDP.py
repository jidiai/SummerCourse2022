#borrow from https://github.com/Grottoh/Deep-Active-Inference-for-Partially-Observable-MDPs


import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import deque, namedtuple

import sys
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))

from train.algo.network import Model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}

class ReplayMemory():

    def __init__(self, capacity, obs_shape, device=DEVICE):

        self.device=device

        self.capacity = capacity # The maximum number of items to be stored in memory

        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity]+[dim for dim in obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)

        self.push_count = 0 # The number of times new data has been pushed to memory

    def push(self, obs, action, reward, done):

        # Store data to memory
        self.obs_mem[self.position()] = obs
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.done_mem[self.position()] = done

        self.push_count += 1

    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity


    def sample(self, obs_indices, action_indices, reward_indices, done_indices, max_n_indices, batch_size):
        # Fine as long as max_n is not greater than the fewest number of time steps an episode can take

        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity)-max_n_indices*2, batch_size, replace=False) + max_n_indices

        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position()+max_n_indices):
                end_indices[i] += max_n_indices

        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index-obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index-action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index-reward_indices for index in end_indices])]
        done_batch = self.done_mem[np.array([index-done_indices for index in end_indices])]

        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.done_mem[index-j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0]) # Reward of 0 will probably not make sense for every environment
                    for k in range(len(done_indices)):
                        if done_indices[k] >= j:
                            done_batch[i, k] = torch.zeros_like(self.done_mem[0])
                    break

        return obs_batch, action_batch, reward_batch, done_batch


class ActiveInference_agent():
    n_hidden_trans = 64
    lr_trans= 1e-3
    n_hidden_pol= 64
    lr_pol= 1e-3
    n_hidden_val= 64
    lr_val= 1e-4
    memory_capacity= 50000
    batch_size= 64
    freeze_period= 25
    beta = 0.99
    gamma = 1
    n_episodes = 10000

    def __init__(self, obs_shape, n_action, device, env, logdir, writer):
        self.obs_shape = obs_shape
        self.obs_size = np.prod(self.obs_shape)
        self.n_actions = n_action
        self.device = device

        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network

        # Initialize the networks:
        self.transition_net = Model(self.obs_size+1, self.obs_size, self.n_hidden_trans, lr=self.lr_trans, device=self.device)
        self.policy_net = Model(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)

        self.target_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())

        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)

        self.obs_indices = [2, 1, 0]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1

        self.env = env
        self.logdir=logdir
        self.writer = writer

    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
            self.obs_indices, self.action_indices, self.reward_indices,
            self.done_indices, self.max_n_indices, self.batch_size)

        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])

        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

        # At time t0 predict the state at time t1:
        X = torch.cat((obs_batch_t0, action_batch_t0.float()), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
            pred_batch_t0t1, obs_batch_t1, reduction='none'), dim=1).unsqueeze(1)

        return (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
                action_batch_t1, reward_batch_t1, done_batch_t2, pred_error_batch_t0t1)


    def compute_value_net_loss(self, obs_batch_t1, obs_batch_t2,
                                   action_batch_t1, reward_batch_t1,
                                   done_batch_t2, pred_error_batch_t0t1):

        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(obs_batch_t2)

            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(obs_batch_t2)

            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1-done_batch_t2) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)

            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.beta * weighted_targets

        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(obs_batch_t1).gather(1, action_batch_t1)

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)

        return value_net_loss

    def compute_VFE(self, obs_batch_t1, pred_error_batch_t0t1):

        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(obs_batch_t1)

        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(obs_batch_t1).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)

        # Weigh them according to the action distribution:
        energy_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.batch_size, 1)

        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).view(self.batch_size, 1)

        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        VFE = torch.mean(VFE_batch)

        return VFE

    def select_action(self, obs):
        with torch.no_grad():
            policy = self.policy_net(obs)
            return torch.multinomial(policy, 1)

    def learn(self):

        # If there are not enough transitions stored in memory, return:
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return

        # After every freeze_period time steps, update the target network:
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1

        # Retrieve transition data in mini batches:
        (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
         action_batch_t1, reward_batch_t1, done_batch_t2,
         pred_error_batch_t0t1) = self.get_mini_batches()

        # Compute the value network loss:
        value_net_loss = self.compute_value_net_loss(obs_batch_t1, obs_batch_t2,
                                                     action_batch_t1, reward_batch_t1,
                                                     done_batch_t2, pred_error_batch_t0t1)

        # Compute the variational free energy:
        VFE = self.compute_VFE(obs_batch_t1, pred_error_batch_t0t1)

        # Reset the gradients:
        self.transition_net.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()

        # Compute the gradients:
        VFE.backward()
        value_net_loss.backward()

        # Perform gradient descent:
        self.transition_net.optimizer.step()
        self.policy_net.optimizer.step()
        self.value_net.optimizer.step()

        self.writer.add_scalar("train/value loss", value_net_loss.mean().item())
        self.writer.add_scalar('train/VFE', VFE.mean().item())

    def update(self):
        record_win = deque(maxlen=100)
        results = []
        for ith_episode in range(self.n_episodes):
            total_reward = 0
            obs = self.env.reset()
            obs = obs[0].flatten()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            done = False
            reward = 0
            while not done:
                action = self.select_action(obs)
                self.memory.push(obs, action, reward, done)


                input_action = actions_map[action[0].item()]
                input_action = [[i for i in input_action]]
                obs, reward, done, info = self.env.step(input_action)
                reward = reward[0]
                obs = obs[0].flatten()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                total_reward += reward

                # self.env.render()
                self.learn()

                if done:
                    self.memory.push(obs, -99, -99, done)    #fixme
                    win_is = 1 if info == 'finished' else 0
                    record_win.append(win_is)

                    print("Episode: ", ith_episode, "; Episode Return: ", total_reward,'; Trained episode:', ith_episode,
                          "win rate = ", sum(record_win)/len(record_win))

                    self.writer.add_scalar('Rollout/Gt', total_reward, ith_episode)


            results.append(total_reward)

            if ith_episode % self.n_episodes == 0:
                self.save(self.logdir, ith_episode)



    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.policy_net.state_dict(), model_actor_path)
        model_transition_path = os.path.join(base_path, "transition_" + str(episode) + ".pth")
        torch.save(self.transition_net.state_dict(), model_transition_path)
        model_value_path = os.path.join(base_path, "value_" + str(episode) + ".pth")
        torch.save(self.value_net.state_dict(), model_value_path)

    def load(self, run_dir, episode):
        raise NotImplementedError
