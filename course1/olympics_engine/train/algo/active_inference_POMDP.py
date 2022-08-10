#borrow from https://github.com/Grottoh/Deep-Active-Inference-for-Partially-Observable-MDPs

import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import deque, namedtuple

import datetime
import sys
import random
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))

from train.algo.network import Model


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}
# actions_map = {0: [0,0], 1:[100, 0], 2: [100,10], 3: [100,-10], 4: [-50,0]}

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

class ReplayMemory():

    def __init__(self, capacity, obs_shape, device='cpu'):

        self.device=device

        self.capacity = capacity # The maximum number of items to be stored in memory

        self.obs_shape = obs_shape # the shape of observations

        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity]+[dim for dim in self.obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)

        self.push_count = 0 # The number of times new data has been pushed to memory

    def push(self, obs, action, reward, done):

        # Store data in memory
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

    def get_last_n_obs(self, n):
        """ Get the last n observations stored in memory (of a single episode) """
        last_n_obs = torch.zeros([n]+[dim for dim in self.obs_shape], device=self.device)

        n = min(n, self.push_count)
        for i in range(1, n+1):
            if self.position() >= i:
                if self.done_mem[self.position()-i]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[self.position()-i]
            else:
                if self.done_mem[-i+self.position()]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[-i+self.position()]

        return last_n_obs


import torch.nn as nn


class VAE(nn.Module):
    # In part taken from:
    #   https://github.com/pytorch/examples/blob/master/vae/main.py

    def __init__(self, obs_shape, n_screens, n_latent_states, lr=1e-5, device = 'cuda', obs_type='flat'):
        super(VAE, self).__init__()

        self.device = device

        self.n_screens = n_screens
        self.n_latent_states = n_latent_states

        self.obs_shape = obs_shape
        self.obs_size = np.prod(obs_shape)

        self.obs_type = obs_type
        if obs_type == 'flat':
            self.encoder_in_size = self.obs_size * n_screens
            self.decoder_out_size = self.obs_size * n_screens
            self.conv3d_shape_out = [self.obs_shape[0], n_screens]
            self.fc1 = nn.Linear(self.encoder_in_size, self.encoder_in_size // 2)
            self.fc2_mu = nn.Linear(self.encoder_in_size // 2, self.n_latent_states)
            self.fc2_logvar = nn.Linear(self.encoder_in_size // 2, self.n_latent_states)

            # Fully connected layers connected to decoder
            self.fc3 = nn.Linear(self.n_latent_states, self.decoder_out_size // 2)
            self.fc4 = nn.Linear(self.decoder_out_size // 2, self.decoder_out_size)
        elif obs_type == 'RGB':

            # The convolutional encoder
            if len(self.obs_shape) >1:
                self.encoder1 = nn.Sequential(
                    nn.Conv3d(3, 16, (5, 5, 1), (2, 2, 1)),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                )
                self.encoder2 = nn.Sequential(
                    nn.Conv3d(16, 32, (5, 5, 1), (2, 2, 1)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                )
                self.encoder3 = nn.Sequential(
                    nn.Conv3d(32, 32, (5, 5, 1), (2, 2, 1)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True)
                ).to(self.device)

            # The size of the encoder output
            self.conv3d_shape_out = (32, 2, 2, self.n_screens)
            self.conv3d_size_out = np.prod(self.conv3d_shape_out)

            # The convolutional decoder
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose3d(32, 32, (5, 5, 1), (2, 2, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose3d(32, 16, (6, 6, 1), (2, 2, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose3d(16, 3, (6, 6, 1), (2, 2, 1)),
                nn.BatchNorm3d(3),
                nn.ReLU(inplace=True),
                nn.Sigmoid()
            ).to(self.device)

            # Fully connected layers connected to encoder
            self.fc1 = nn.Linear(self.conv3d_size_out, self.conv3d_size_out // 2)
            self.fc2_mu = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
            self.fc2_logvar = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
            self.fc3 = nn.Linear(self.n_latent_states, self.conv3d_size_out // 2)
            self.fc4 = nn.Linear(self.conv3d_size_out // 2, self.conv3d_size_out)

        else:
            raise NotImplementedError

        self.optimizer = torch.optim.Adam(self.parameters(), lr)

        self.to(self.device)

    def encode(self, x):
        # Deconstruct input x into a distribution over latent states
        if self.obs_type == 'flat':
            conv = x  #self.encoder(x)
        elif self.obs_type == 'RGB':
            conv1 = self.encoder1(x)  #18,18
            conv2 = self.encoder2(conv1)  #7,7
            conv = self.encoder3(conv2)  #2,2
        else:
            raise NotImplementedError

        h1 = F.relu(self.fc1(conv.reshape(conv.size(0), -1)))
        mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Apply reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, batch_size=1):
        # Reconstruct original input x from the (reparameterized) latent states
        h3 = F.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view([batch_size] + [dim for dim in self.conv3d_shape_out])
        if self.obs_type == 'flat':
            y = deconv_input
        elif self.obs_type == 'RGB':
            y1 = self.decoder1(deconv_input)  #7,7
            y2 = self.decoder2(y1)  #17,17
            y = self.decoder3(y2)# 37, 37
        else:
            raise NotImplementedError

        return y

    def forward(self, x, batch_size=1):
        # Deconstruct and then reconstruct input x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, batch_size)
        return recon, mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, batch=True):
        if batch:
            # BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
            # BCE = torch.sum(BCE, dim=(1, 2))
            if self.obs_type == 'flat':
                l = F.mse_loss(recon_x, x, reduction='none')
                l = torch.sum(l, dim = (1,2))
            elif self.obs_type == 'RGB':
                l = F.binary_cross_entropy(recon_x, x, reduction='none')
                l = torch.sum(l, dim=(1, 2,3,4))

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            if self.obs_type == 'flat':
                l = F.mse_loss(recon_x, x, reduction='sum')
            elif self.obs_type == 'RGB':
                l = F.binary_cross_entropy(recon_x, x, reduction='sum')

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return l + KLD


# import torchvision.transforms as T
from pathlib import Path
current_path =  str(Path(__file__).resolve().parent)


class ActiveInference_POMDP_agent():
    n_screens = 4
    n_latent_states = 32
    lr_vae = 1e-5
    alpha=25000
    n_hidden_trans=64
    lr_trans = 1e-3
    n_hidden_pol=64
    lr_pol=0.001
    n_hidden_val=64
    lr_val=1e-4
    memory_capacity = 50000
    batch_size=32
    freeze_period = 25
    beta=0.99
    e_lambda = 0.1
    gamma = 12
    obs_preprocess_type = 'flat'  #flat, onehot, RGB

    pre_train_vae = True
    load_vae_path = ''
    pt_vae_n_episodes = 5000
    n_episodes = 5000

    keep_log = True
    render = False
    vae_saving_interval = 500
    model_saving_interval = 100
    print_timer = 50


    def __init__(self, obs_shape, n_action, device, env, logdir, writer):


        self.freeze_cntr = 0
        self.device = device
        self.n_actions = n_action
        if self.obs_preprocess_type == 'flat':
            self.obs_shape = [np.prod(obs_shape)]
        elif self.obs_preprocess_type == 'onehot':
            self.obs_shape = obs_shape + [11]
        elif self.obs_preprocess_type == 'RGB':
            self.obs_shape = obs_shape + [3]

        self.env = env
        self.logdir=logdir
        if not os.path.exists(os.path.join(self.logdir, 'networks/pre_trained_vae')):
            os.makedirs(os.path.join(self.logdir, 'networks/pre_trained_vae'))
        if not os.path.exists(os.path.join(self.logdir, 'networks/trained_model')):
            os.makedirs(os.path.join(self.logdir, 'networks/trained_model'))


        self.writer = writer

        self.obs_preprocess_fn = self.get_preprocessor(self.obs_preprocess_type)

        # Initialize the networks:
        self.vae = VAE(self.obs_shape, self.n_screens, self.n_latent_states, lr=self.lr_vae, device=self.device, obs_type = self.obs_preprocess_type)
        self.transition_net = Model(self.n_latent_states*2+1, self.n_latent_states, self.n_hidden_trans, lr=self.lr_trans, device=self.device)
        self.policy_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())


        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)

        if self.load_vae_path != '':
            loaded_vae = torch.load(self.load_vae_path, map_location=self.device)
            self.vae.load_state_dict(loaded_vae)


        # self.resize = T.Compose([T.ToPILImage(),
        #                          T.Resize(40, interpolation=Image.CUBIC),
        #                          T.ToTensor()])

        self.obs_indices = [(self.n_screens+1)-i for i in range(self.n_screens+2)]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1

        self.total_training_cnt = 0

    def get_preprocessor(self, name):
        if name == 'flat':
            return ActiveInference_POMDP_agent.flatten
        elif name == 'onehot':
            return ActiveInference_POMDP_agent.onehot_obs
        elif name == 'RGB':
            return ActiveInference_POMDP_agent.RGB_obs
        else:
            raise NotImplementedError

    @staticmethod
    def flatten(obs):
        return obs.flatten()

    @staticmethod
    def onehot_obs(obs, n=11):
        # encoded_obs = np.zeros((n,n,11))
        encoded_obs = (np.arange(n)==obs[...,None]-1).astype(int)
        return encoded_obs

    @staticmethod
    def RGB_obs(obs):
        n = obs.shape[0]
        encoded_obs = np.zeros((n,n,3))
        for row in range(n):
            for col in range(n):
                encoded_obs[row, col, ...] = COLORS[IDX_TO_COLOR[obs[row, col]]]
        return encoded_obs/255


    def select_action(self, obs):
        with torch.no_grad():
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            # Derive a distribution over states state from the last n observations (screens):
            prev_n_obs = self.memory.get_last_n_obs(self.n_screens-1)

            if self.obs_preprocess_type == 'flat':
               review = (1,-1,self.n_screens)
            elif self.obs_preprocess_type == 'RGB':
                review = (1, self.obs_shape[-1], self.obs_shape[0], self.obs_shape[1], self.n_screens)

            x = torch.cat((prev_n_obs, obs), dim=0).reshape(review)
            state_mu, state_logvar = self.vae.encode(x)

            # Determine a distribution over actions given the current observation:
            x = torch.cat((state_mu, torch.exp(state_logvar)), dim=1)
            policy = self.policy_net(x)
            return torch.multinomial(policy, 1)


    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
            self.obs_indices, self.action_indices, self.reward_indices,
            self.done_indices, self.max_n_indices, self.batch_size)

        if self.obs_preprocess_type == 'flat':
            review = (self.batch_size, self.obs_shape[0], self.n_screens)
        elif self.obs_preprocess_type == 'onehot':
            review = (self.batch_size, self.obs_shape[-1], self.obs_shape[0], self.obs_shape[1], self.n_screens)
        elif self.obs_preprocess_type == 'RGB':
            review = (self.batch_size, self.obs_shape[-1], self.obs_shape[0], self.obs_shape[1], self.n_screens)
        else:
            raise NotImplementedError



        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, ...].reshape(review)  #.view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, ...].reshape(review)  #.view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, ...].reshape(review)            #.view(self.batch_size, self.c, self.h, self.w, self.n_screens)

        # Retrieve a batch of distributions over states for 3 consecutive points in time
        state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
        state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
        state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)

        # Combine the sufficient statistics (mean and variance) into a single vector
        state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
        state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
        state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)

        # Reparameterize the distribution over states for time t1
        z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)

        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

        # At time t0 predict the state at time t1:
        X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
            pred_batch_t0t1, state_mu_batch_t1, reduction='none'), dim=1).unsqueeze(1)

        return (state_batch_t1, state_batch_t2, action_batch_t1,
                reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
                obs_batch_t1, state_mu_batch_t1,
                state_logvar_batch_t1, z_batch_t1)

    def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
                               action_batch_t1, reward_batch_t1,
                               done_batch_t2, pred_error_batch_t0t1):

        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(state_batch_t2)

            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(state_batch_t2)

            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1-done_batch_t2) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)

            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.beta * weighted_targets

        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(state_batch_t1).gather(1, action_batch_t1)

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)

        return value_net_loss

    def compute_VFE(self, vae_loss, state_batch_t1, pred_error_batch_t0t1):

        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(state_batch_t1)

        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(state_batch_t1).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)

        # Weigh them according to the action distribution:
        energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)

        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)
        # print('policy batch = ', torch.log(policy_batch_t1).shape)

        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = vae_loss + pred_error_batch_t0t1 + (energy_term_batch - self.e_lambda*entropy_batch)
        VFE = torch.mean(VFE_batch)

        return VFE, entropy_batch.detach().cpu().mean()

    def learn(self):

        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return

        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1

        # Retrieve mini-batches of data from memory
        (state_batch_t1, state_batch_t2, action_batch_t1,
         reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
         obs_batch_t1, state_mu_batch_t1,
         state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()

        # Determine the reconstruction loss for time t1
        recon_batch = self.vae.decode(z_batch_t1, self.batch_size)
        vae_loss = self.vae.loss_function(recon_batch, obs_batch_t1, state_mu_batch_t1, state_logvar_batch_t1, batch=True) / self.alpha

        # Compute the value network loss:
        value_net_loss = self.compute_value_net_loss(state_batch_t1, state_batch_t2,
                                                     action_batch_t1, reward_batch_t1,
                                                     done_batch_t2, pred_error_batch_t0t1)

        # Compute the variational free energy:
        VFE, entropy = self.compute_VFE(vae_loss, state_batch_t1.detach(), pred_error_batch_t0t1)

        # Reset the gradients:
        self.vae.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.transition_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()

        # Compute the gradients:
        VFE.backward(retain_graph=True)
        value_net_loss.backward()

        policy_fc1_grad = self.policy_net.fc1.weight.grad.mean().item()
        policy_fc2_grad = self.policy_net.fc2.weight.grad.mean().item()
        value_fc1_grad = self.value_net.fc1.weight.grad.mean().item()
        value_fc2_grad = self.value_net.fc2.weight.grad.mean().item()


        # Perform gradient descent:
        self.vae.optimizer.step()
        self.policy_net.optimizer.step()
        self.transition_net.optimizer.step()
        self.value_net.optimizer.step()

        self.total_training_cnt += 1
        self.writer.add_scalar("learn/VAE loss", vae_loss.mean().cpu().item(), self.total_training_cnt)
        self.writer.add_scalar('learn/value loss', value_net_loss.cpu().item(), self.total_training_cnt)
        self.writer.add_scalar('learn/VFE loss', VFE.cpu().item(), self.total_training_cnt)
        self.writer.add_scalar('learn/action entropy', entropy.item(), self.total_training_cnt)

        self.writer.add_scalar('Gradient/policy_fc1', policy_fc1_grad, self.total_training_cnt)
        self.writer.add_scalar('Gradient/policy_fc2', policy_fc2_grad, self.total_training_cnt)
        self.writer.add_scalar('Gradient/value_fc1', value_fc1_grad, self.total_training_cnt)
        self.writer.add_scalar('Gradient/value_fc2', value_fc2_grad, self.total_training_cnt)

    def train_vae(self):
        """ Train the VAE separately. """

        vae_batch_size = 256
        vae_obs_indices = [self.n_screens-i for i in range(self.n_screens)]
        train_cnt = 0
        losses = []
        for ith_episode in range(self.pt_vae_n_episodes):

            obs = self.env.reset()[0]
            obs = self.obs_preprocess_fn(obs)
            done = False
            while not done:

                action = random.randint(0, len(actions_map)-1)  #self.env.action_space.sample()
                obs = torch.tensor(obs, dtype=torch.float32, device = self.device)
                self.memory.push(obs, -99, -99, done)

                input_action = actions_map[action]
                input_action = [[i for i in input_action]]

                obs, reward, done, info = self.env.step(input_action)
                obs = self.obs_preprocess_fn(obs[0])
                if self.render:
                    self.env.render()
                reward = reward[0]

                # obs = obs[0].flatten()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

                if self.memory.push_count > vae_batch_size + self.n_screens*2:
                    obs_batch, _, _, _ = self.memory.sample(vae_obs_indices, [], [], [], len(vae_obs_indices), vae_batch_size)
                    # obs_batch = obs_batch.view(vae_batch_size, -1, self.n_screens)
                    obs_batch = obs_batch.transpose(1, -1)          #[batch, channel, height, width, n_screen]
                    # obs_batch = obs_batch.view(vae_batch_size, self.c, self.h, self.w, self.n_screens)

                    recon, mu, logvar = self.vae.forward(obs_batch, vae_batch_size)
                    loss = torch.mean(self.vae.loss_function(recon, obs_batch, mu, logvar))

                    self.vae.optimizer.zero_grad()
                    loss.backward()
                    self.vae.optimizer.step()

                    losses.append(loss)
                    self.writer.add_scalar("pretrained VAE/loss", loss.item(), train_cnt)
                    train_cnt += 1
                    if train_cnt % 100 == 0:
                        print("episode %4d: vae_loss=%5.2f"%(ith_episode, loss.item()))

                    # if done:
                    #     if ith_episode > 0 and ith_episode % 10 > 0 and self.pt_vae_plot:
                    #         plt.plot(losses)
                    #         plt.show()
                    #         plt.plot(losses[-1000:])
                    #         plt.show()
                    #         for i in range(self.n_screens):
                    #             plt.imshow(obs_batch[0, :, :, :, i].detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                    #             plt.show()
                    #             plt.imshow(recon[0, :, :, :, i].detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                    #             plt.show()

                if done:
                    self.memory.push(obs, -99, -99, done)

                    if ith_episode > 0 and ith_episode % self.vae_saving_interval == 0:
                        # if not os.path.exists(os.path.join(self.logdir, 'networks/pre_trained_vae')):
                        #     os.mkdir(os.path.join(self.logdir, 'networks/pre_trained_vae'))
                        vae_path = os.path.join(self.logdir, "networks/pre_trained_vae/vae_{}_n{}_{:d}.pth".format(
                            self.obs_preprocess_type,
                            self.n_latent_states, ith_episode))
                        torch.save(self.vae.state_dict(), vae_path)
                        print(f'Vae model saved at episode {ith_episode}')

        self.memory.push_count = 0
        vae_path=os.path.join(self.logdir, "networks/pre_trained_vae/vae_{}_n{}_end.pth".format(
            self.obs_preprocess_type,self.n_latent_states))
        torch.save(self.vae.state_dict(), vae_path)

    def update(self):

        if self.pre_train_vae: # If True: pre-train the VAE
            msg = "##### Pre-training vae. Starting at {}".format(datetime.datetime.now())
            print(msg)
            self.train_vae()

        total_rollout_cnt = 0
        # msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        # print(msg)
        # if self.keep_log:
        #     self.record.write(msg+"\n")

        results = []
        for ith_episode in range(self.n_episodes):

            total_reward = 0
            obs = self.env.reset()[0]
            obs = self.obs_preprocess_fn(obs)
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            # obs = self.get_screen(self.env, self.device)
            if self.render:
                self.env.render()

            done = False
            reward = 0

            while not done:

                action = self.select_action(obs)
                self.memory.push(obs, action, reward, done)

                input_action = actions_map[action[0].item()]
                input_action = [[i for i in input_action]]

                obs, reward, done, info = self.env.step(input_action)
                reward = reward[0]
                obs = self.obs_preprocess_fn(obs[0])
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

                if self.render:
                    self.env.render()

                total_reward += reward

                self.learn()

                if done:
                    self.memory.push(obs, -99, -99, done)
                    total_rollout_cnt += 1
                    self.writer.add_scalar("rollout/total reward", total_reward, total_rollout_cnt)

            results.append(total_reward)

            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                avg_reward = np.mean(results)
                last_x = np.mean(results[-self.print_timer:])
                msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
                print(msg)

                # if self.keep_log:
                #     self.record.write(msg+"\n")
                #
                #     if ith_episode % self.log_save_timer == 0:
                #         self.record.close()
                #         self.record = open(self.log_path, "a")

            if ith_episode > 0 and ith_episode % self.model_saving_interval == 0:
                # if not os.path.exists(os.path.join(self.logdir, 'networks/pre_trained_vae')):
                #     os.mkdir(os.path.join(self.logdir, 'networks/pre_trained_vae'))
                model_path = os.path.join(self.logdir, "networks/trained_model")
                torch.save(self.value_net.state_dict(), os.path.join(model_path, f"value_{ith_episode}.pth"))
                torch.save(self.vae.state_dict(), os.path.join(model_path, f"VAE_{ith_episode}.pth"))
                torch.save(self.policy_net.state_dict(), os.path.join(model_path, f'policy_{ith_episode}.pth'))
                torch.save(self.transition_net.state_dict(), os.path.join(model_path, f'transition_{ith_episode}.pth'))
                print(f'model saved at episode {ith_episode}')




            # If enabled, save the results and the network (state_dict)
            # if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
            #     np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
            # if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
            #     torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_networks{}_{:d}.pth".format(self.run_id, ith_episode))

        self.env.close()

        # If enabled, save the results and the network (state_dict)
        # if self.save_results:
        #     np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
        #     np.savez(self.results_path, np.array(results))
        # if self.save_network:
        #     torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_networks{}_end.pth".format(self.run_id))
        #     torch.save(self.value_net.state_dict(), self.network_save_path)
        #
        # # Print and keep a (.txt) record of stuff
        # msg = "Training finished at {}".format(datetime.datetime.now())
        # print(msg)
        # if self.keep_log:
        #     self.record.write(msg)
        #     self.record.close()