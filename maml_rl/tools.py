import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from gym.spaces.box import Box
from bc_gym_planning_env.envs.base import spaces
from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.envs.egocentric import EgocentricCostmap
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples

from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv

from gym_wrapper import bc_gym_wrapper

from collections import OrderedDict

from torch.distributions import Normal

from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

import torch.multiprocessing as mp
import copy
import os

class BatchEpisodes(object):
    def __init__(self, batch_size, device, gamma=0.95):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observation, action, reward, batch_id):
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(np.float32(reward))

    def __len__(self):
        return max(map(len, self._rewards_list))
        
class Sampler(object):
    def __init__(self, env_name, meta_iter, batch_size, device):
        if env_name == 'RandomMiniEnv':
            self.env = RandomMiniEnv
        elif env_name == 'RandomAisleTurnEnv':
            self.env = RandomAisleTurnEnv
    
        self.meta_iter = meta_iter
        self.batch_size = batch_size        
        self.device = device

    def sample_tasks(self, low, high, num_tasks):
        # seeds = np.random.randint(low=low, high=high, size=num_tasks)
        seeds = [5, 44, 122, 134, 405, 587, 1401, 1408, 1693, 1796]
        
        #validation_seeds = [2262, 2302, 4151, 2480, 2628]
        
        tasks = []

        env_param = EnvParams(iteration_timeout=300,
                              goal_ang_dist=np.pi/8,
                              goal_spat_dist=1.0, # was 0.2
                              robot_name=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1)

        for s in seeds:
            env = EgocentricCostmap(self.env(params=env_param, 
                                             turn_off_obstacles=False,
                                             draw_new_turn_on_reset=False,
                                             seed=s))
            env = bc_gym_wrapper(env, normalize=True)
            
            tasks.append(env)
        
        return tasks

    
    def generate_episodes(self, task, policy, num_episodes):        
        episodes = BatchEpisodes(batch_size=self.batch_size, device=self.device)
        
        traj_id = 0
        
        done = False
        state = task.reset()
        
        
        while not done:
            with torch.no_grad():
                action = policy(torch.Tensor(state).to(device=self.device)).sample()
                action = action.cpu().numpy()
                
            next_state, reward, done, _ = task.step(np.clip(action, -1.0, 1.0))
            episodes.append(next_state, action, reward, traj_id)
            
            state = next_state
            
            if done:
                traj_id += 1

                if traj_id == num_episodes:
                    return episodes

                done = False
                state = task.reset()

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
            
class Policy(nn.Module):
    def __init__(self, init_std=1.0, min_std=1e-6):
        super(Policy, self).__init__()
        self.min_log_std = math.log(min_std)
                
        self.fc1 = nn.Linear(135, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.tanh = nn.Tanh()
        
        self.sigma = nn.Parameter(torch.Tensor(2))
        self.sigma.data.fill_(math.log(init_std))
        
        self.apply(weight_init)
        
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
                    
        output = F.relu(F.linear(input, weight=params['fc1.weight'], bias=params['fc1.bias']))
        output = F.relu(F.linear(output, weight=params['fc2.weight'], bias=params['fc2.bias']))
        output = F.linear(output, weight=params['fc3.weight'], bias=params['fc3.bias'])
        output = self.tanh(output)
        
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        
        return Normal(loc=output, scale=scale)
    
    
    def update_params(self, loss, step_size=0.5, first_order=False):
        grads = torch.autograd.grad(loss, self.parameters(),
            create_graph=not first_order)
        
        updated_params = OrderedDict()
        
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params
    
class LinearFeatureBaseline(nn.Module):
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations * ones
        cum_sum = torch.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.0

        return torch.cat([observations, observations ** 2,
            al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)

        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32,
            device=self.linear.weight.device)

        for _ in range(5):
            try:
                arr1 = torch.matmul(featmat.t(), returns).cpu().numpy()
                arr2 = (torch.matmul(featmat.t(), featmat) + reg_coeff * eye).cpu().numpy()
                
                coeffs, _, _, _ = np.linalg.lstsq(arr1, arr2, rcond=1)
                coeffs = torch.Tensor(coeffs).to(self.linear.weight.device)
                coeffs = torch.transpose(coeffs, 0, 1)
                
                break
            except RuntimeError:
                reg_coeff += 10
        else:
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.linear.weight.data = coeffs.data.t()

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)
    
def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()

def weighted_mean(tensor, dim=None, weights=None):
    if weights is None:
        out = torch.mean(tensor)
    if dim is None:
        out = torch.sum(tensor * weights)
        out.div_(torch.sum(weights))
    else:
        mean_dim = torch.sum(tensor * weights, dim=dim)
        mean_dim.div_(torch.sum(weights, dim=dim))
        out = torch.mean(mean_dim)
        
    return out

def weighted_normalize(tensor, dim=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, dim=dim, weights=weights)
    out = tensor * (1 if weights is None else weights) - mean
    std = torch.sqrt(weighted_mean(out ** 2, dim=dim, weights=weights))
    out.div_(std + epsilon)
    
    return out

def detach_distribution(pi):
    distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())

    return distribution

class MetaLearner(object):
    def __init__(self, sampler, policy, baseline, num_episodes, gamma=0.95,
                 fast_lr=0.01, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.num_episodes = num_episodes

        self.device = device
        
        self.to(device)

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=episodes.mask)

        return loss

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=first_order)

        return params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        
        episodes = []
        for i, task in enumerate(tasks):                                                                            
            train_episodes = self.sampler.generate_episodes(task, self.policy, num_episodes=self.num_episodes)
                        
            params = self.adapt(train_episodes, first_order=first_order)
            
            valid_episodes = self.sampler.generate_episodes(task, self.policy, num_episodes=self.num_episodes)
                        
            episodes.append((train_episodes, valid_episodes))
                        
        return episodes
    
    def average_return(self, episodes):
        total_return = 0
        
        for _, valid_episodes in episodes:
            total_return += torch.sum(valid_episodes.rewards)
                    
        ret = total_return.item()/(len(episodes) * self.sampler.batch_size)
        
        return ret


    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-2, cg_iters=10, cg_damping=1e-5,
             ls_max_steps=15, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        
        loss = old_loss.item()
        
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
            
        return loss

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
