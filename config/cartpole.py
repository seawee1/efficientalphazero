from copy import deepcopy
from typing import Optional
import numpy as np
import torch.nn as nn
import gym
from gym.spaces import Discrete

from config.base import BaseConfig
from core.model import StandardModel
from core.util import DiscreteSupport


class CartpoleWithTime(gym.Env):
    def __init__(self, config=None):
        self.env = gym.make("CartPole-v1")

        self.action_space = Discrete(2)
        s = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.array(s.low.tolist() + [0.0], dtype=float),
            high=np.array(s.high.tolist() + [1.0], dtype=float),
            shape=(s.shape[0] + 1,)
        )

        self.time_step = 0
        self.past_actions = []

    def reset(self, **kwargs):
        self.past_actions = []
        self.time_step = 0

        obs = self.env.reset()

        obs_ = np.zeros((self.observation_space.shape[0],), dtype=np.float32)
        obs_[:obs.shape[0]] = obs
        obs_[-1] = self.time_step / 500

        return obs_

    def step(self, action):
        obs, rew, terminated, info = self.env.step(action)
        self.time_step += 1

        obs_ = np.zeros((self.observation_space.shape[0],), dtype=np.float32)
        obs_[:obs.shape[0]] = obs
        obs_[-1] = self.time_step / 500

        score = 0.0 if terminated else rew

        self.past_actions.append(action)
        return (
            obs_,
            score,
            terminated,
            info,
        )

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def set_state(self, state):
        self.env = deepcopy(state[0])
        self.time_step = deepcopy(state[1])
        self.past_actions = deepcopy(state[2])
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env), deepcopy(self.time_step), deepcopy(self.past_actions)


class Config(BaseConfig):
    def __init__(
            self,
            training_steps: int = 20,
            pretrain_steps: int = 0,
            model_broadcast_interval: int = 5,
            num_sgd_iter: int = 10,
            clear_buffer_after_broadcast: bool = False,
            root_value_targets: bool = False,
            replay_buffer_size: int = 50000,
            demo_buffer_size: int = 0,
            batch_size: int = 512,
            lr: float = 0.2,
            max_grad_norm: float = 5,
            weight_decay: float = 1e-4,
            momentum: float = 0.9,
            c_init: float = 3.0,
            c_base: float = 19652,
            gamma: float = 0.997,
            frame_stack: int = 5,
            max_reward_return: bool = False,
            hash_nodes: bool = False,
            root_dirichlet_alpha: float = 1.5,
            root_exploration_fraction: float = 0.25,
            num_simulations: int = 30,
            num_envs_per_worker: int = 1,
            min_num_episodes_per_worker: int = 2,
            use_dirichlet: bool = True,
            test_use_dirichlet: bool = False,
            value_support: DiscreteSupport = DiscreteSupport(0, 22, 1.0),
            value_transform: bool = True,
            env_seed: int = None,
    ):
        super().__init__(
            training_steps,
            pretrain_steps,
            model_broadcast_interval,
            num_sgd_iter,
            clear_buffer_after_broadcast,
            root_value_targets,
            replay_buffer_size,
            demo_buffer_size,
            batch_size,
            lr,
            max_grad_norm,
            weight_decay,
            momentum,
            c_init,
            c_base,
            gamma,
            frame_stack,
            max_reward_return,
            hash_nodes,
            root_dirichlet_alpha,
            root_exploration_fraction,
            num_simulations,
            num_envs_per_worker,
            min_num_episodes_per_worker,
            use_dirichlet,
            test_use_dirichlet,
            value_support,
            value_transform,
            env_seed
        )

    def create_model(self, device, amp):
        probe_env = self.env_creator()
        obs_shape = probe_env.observation_space.shape
        num_act = probe_env.action_space.n

        model = StandardModel(self, obs_shape, num_act, device, amp)
        size = 512
        model.shared = nn.Sequential(
            nn.Linear(obs_shape[0] * self.frame_stack, size), nn.ReLU(),
            nn.Linear(size, size), nn.ReLU(),
        )
        model.actor = nn.Sequential(
            nn.Linear(size, size), nn.ReLU(),
            nn.Linear(size, size), nn.ReLU(),
            nn.Linear(size, num_act)
        )
        model.critic = nn.Sequential(
            nn.Linear(size, size), nn.ReLU(),
            nn.Linear(size, size), nn.ReLU(),
            nn.Linear(size, self.value_support.size)
        )
        model.to(device)
        return model

    def env_creator(self):
        return CartpoleWithTime()

