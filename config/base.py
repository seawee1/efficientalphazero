import torch
from copy import deepcopy
import gym

from core.util import DiscreteSupport


class BaseConfig:
    def __init__(
            self,
            training_steps: int,  # Number of training steps
            pretrain_steps: int,  # Number of pretraining steps (only supported if Config implements `demonstrator_env_creator` and `collect_demonstration` methods.
            model_broadcast_interval: int,  # When to broadcast updated model weights to rollout workers
            num_sgd_iter: int,  # Number of SGD iterations per update
            clear_buffer_after_broadcast: bool,  # Whether to clear experience buffer after broadcasting new model weights
            root_value_targets: bool,  # Set to `True` to use MCTS value estimate as value target. Otherwise we use episode returns.
            replay_buffer_size: int,
            demo_buffer_size: int,  # Number of demonstration transitions to collect
            batch_size: int,
            lr: float,
            max_grad_norm: float,
            weight_decay: float,
            momentum: float,
            c_init: float,  # Used for PUCT formular
            c_base: float,
            gamma: float,
            frame_stack: int,  # How many frames to stack. Set to 1 for no frame stacking to happen.
            max_reward_return: bool,  # Use max rewards instead of cumsum reward return formulation
            hash_nodes: bool,  # Allows to hash to MCTS tree nodes based on the return environment state. Requires implementation of `hash_env_state` method.
            root_dirichlet_alpha: float,  # Dirichlet noise used for exploration
            root_exploration_fraction: float,
            num_simulations: int,  # Number of simulations per MCTS step
            num_envs_per_worker: int,  # Number of parallel envs per worker. Set to >1 for Batch MCTS (parallel model inference)
            min_num_episodes_per_worker: int,  # Number of episodes each worker has to collect before model weights are updated
            use_dirichlet: bool,  # Whether to use Dirichlet noise during rollouts
            test_use_dirichlet: bool,  # Whether to use Dirichlet noise during testing
            value_support: DiscreteSupport,  # See muZero paper
            value_transform: bool,  # See muZero paper
            env_seed: int,
    ):
        self.training_steps = training_steps
        self.pretrain_steps = pretrain_steps
        self.model_broadcast_interval = model_broadcast_interval
        self.num_sgd_iter = num_sgd_iter
        self.clear_buffer_after_broadcast = clear_buffer_after_broadcast
        self.root_value_targets = root_value_targets

        self.replay_buffer_size = replay_buffer_size
        self.demo_buffer_size = demo_buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.c_init = c_init
        self.c_base = c_base
        self.gamma = gamma
        self.frame_stack = frame_stack
        self.max_reward_return = max_reward_return
        self.hash_nodes = hash_nodes
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.num_simulations = num_simulations

        self.num_envs_per_worker = num_envs_per_worker
        self.min_num_episodes_per_worker = min_num_episodes_per_worker
        self.use_dirichlet: bool = use_dirichlet

        self.test_use_dirichlet = test_use_dirichlet

        self.value_support = value_support
        self.value_transform = value_transform

        self.env_seed = env_seed

        self._action_shape = None
        self._obs_shape = None

    @property
    def obs_shape(self):
        if self._obs_shape is None:
            probe_env = self.env_creator()
            self._obs_shape = probe_env.observation_space.shape
        return self._obs_shape

    @property
    def action_shape(self):
        if self._action_shape is None:
            probe_env = self.env_creator()
            self._action_shape = probe_env.action_space.n
        return self._action_shape

    def env_creator(self):
        raise NotImplementedError

    def demonstrator_env_creator(self):
        raise NotImplementedError

    def collect_demonstration(self, env: gym.Env):
        raise NotImplementedError

    def create_model(self, device):
        raise NotImplementedError

    def hash_env_state(self, env_state):
        raise NotImplementedError

    def phi_transform(self, x: torch.Tensor) -> torch.Tensor:
        # x == 4
        assert len(x.shape) == 1
        x = x.reshape(-1, 1)
        delta = self.value_support.delta

        x.clamp_(self.value_support.min, self.value_support.max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = x - x_low
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], self.value_support.size).to(x.device)
        x_low_idx = (x_low - self.value_support.min) / delta
        x_high_idx = (x_high - self.value_support.min) / delta
        target.scatter_(1, x_low_idx.long()[p_low != 0].reshape(-1, 1), p_low[p_low != 0].reshape(-1, 1))
        target.scatter_(1, x_high_idx.long()[p_high != 0].reshape(-1, 1), p_high[p_high != 0].reshape(-1, 1))
        return target

    def phi_inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2
        r = torch.Tensor(deepcopy(self.value_support.range)).to(x.device)
        return torch.sum(x * r, dim=1)

    def scalar_transform(self, x: torch.Tensor):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        assert len(x.shape) == 1
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
        return output

    def inverse_scalar_transform(self, x: torch.Tensor):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        assert len(x.shape) == 1
        epsilon = 0.001

        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)

        nan_part = torch.isnan(output)
        output[nan_part] = 0.
        output[torch.abs(output) < epsilon] = 0.
        return output


