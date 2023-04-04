import qiskit
from copy import copy
from dataclasses import dataclass, fields, field
from statistics import mean, median
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from scipy.stats import entropy
import torch
import ray


@dataclass
class TrainingBatch:
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    mcts_policies: np.ndarray
    value_targets: np.ndarray
    infos: List[Dict[Any, Any]]

    def to_torch(self, device):
        for f in fields(self):
            if f.name == 'infos':
                continue
            setattr(self, f.name, torch.from_numpy(copy(getattr(self, f.name))).float().to(device))

    def fuse_inplace(self, other_batch):
        for f in fields(self):
            if f.name == 'infos':
                setattr(self, f.name, getattr(self, f.name) + getattr(other_batch, f.name))
            else:
                setattr(self, f.name, np.concatenate((getattr(self, f.name), getattr(other_batch, f.name)), axis=0))


class MCTSRollingWindow:
    def __init__(self, obs_shape, frame_stack):
        self.obs_shape = obs_shape
        self.frame_stack = frame_stack
        self.obs = None
        self.actions = None
        self.rewards = None
        self.env_state = None
        self.info = None
        self.reset()

    def reset(self):
        self.obs = np.zeros(self.obs_shape[0] * self.frame_stack)
        self.actions = np.ones(self.frame_stack) * -1
        self.rewards = np.zeros(self.frame_stack)
        self.env_state = None
        self.info = None

    def add(self, obs, env_state, reward=None, action=None, info=None):
        self.obs = np.roll(self.obs, self.obs_shape[0])
        self.env_state = env_state
        self.info = info
        self.obs[:self.obs_shape[0]] = obs
        if reward is not None:
            self.rewards = np.roll(self.rewards, 1)
            self.rewards[0] = reward
        if action is not None:
            self.actions = np.roll(self.actions, 1)
            self.actions[0] = action

    def latest_obs(self):
        return self.obs.reshape(self.frame_stack, -1)[0]


@dataclass
class TransitionBuffer:
    obs: List[np.ndarray] = field(default_factory=lambda: [])
    rewards: List[float] = field(default_factory=lambda: [])
    dones: List[bool] = field(default_factory=lambda: [])
    infos: List[Dict[str, Any]] = field(default_factory=lambda: [])
    mcts_policies: List[np.ndarray] = field(default_factory=lambda: [])
    value_targets: List[Optional[float]] = field(
        default_factory=lambda: [])  # Can be list of `None` until end of episode
    env_states: List[Dict[str, Any]] = field(default_factory=lambda: [])
    priorities: List[float] = field(default_factory=lambda: [])

    def extend(self, sample_batch: 'TransitionBuffer') -> None:
        for f in fields(self):
            getattr(self, f.name).extend(getattr(sample_batch, f.name))

    def augment_value_targets(self, accum):
        ret = 0.0
        for i in reversed(range(self.size())):
            self.value_targets[i] = ret
            ret = accum([ret, self.rewards[i]])

    def add_one(self, obs, reward, done, info, mcts_policy, value_target, env_state, priority):
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.mcts_policies.append(mcts_policy)
        self.value_targets.append(value_target)
        self.env_states.append(env_state)
        self.priorities.append(priority)

    def frame_stack_at_index(self, index, field_name, frame_stack, pad_value=0.0):
        vals = getattr(self, field_name)
        is_array = isinstance(getattr(self, field_name)[0], np.ndarray)
        frame = np.empty(0) if is_array else []

        pad = False
        while frame_stack > 0:
            if pad:
                if is_array:
                    frame = np.concatenate((frame, np.ones_like(vals[0]) * pad_value))
                else:
                    frame.append(pad_value)
            else:
                val = vals[index]
                if is_array:
                    frame = np.concatenate((frame, val))
                else:
                    frame.append(val)
            index -= 1
            frame_stack -= 1
            if index < 0 or self.dones[index]:
                pad = True

        return frame

    def size(self) -> int:
        return len(self.obs)

    def drop_first_n(self, n: int) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name)[n:])

    def sample_training_batch(self, batch_size: int, frame_stack: int) -> Tuple[TrainingBatch, np.ndarray]:
        if batch_size > self.size():
            print("Warning: trying to sample more samples than in transition buffer!")

        alpha = 1.0
        probs = np.array(self.priorities) ** alpha
        probs /= probs.sum()
        sample_indices = np.random.choice(range(self.size()), min(batch_size, self.size()), p=probs)

        train_batch_args = []
        for f in fields(TrainingBatch):
            field_name = f.name
            samples = getattr(self, field_name)  # [sample_indices]
            if field_name == 'obs':  # We frame-stack obs
                samples = list(map(lambda i: self.frame_stack_at_index(i, 'obs', frame_stack), sample_indices))
            else:
                samples = list(map(lambda i: samples[i], sample_indices))

            if field_name != 'infos':
                samples = np.array(samples)

            train_batch_args.append(samples)

        return TrainingBatch(*train_batch_args), sample_indices

    @staticmethod
    def fuse_buffers(buffers: List['TransitionBuffer']):
        fused_buffer = TransitionBuffer()
        for part_buffer in buffers:
            fused_buffer.extend(part_buffer)
        return fused_buffer

    def get_stats(self):
        stats = {}
        stats['eps_reward'] = sum(self.rewards)
        stats['step_reward_mean'] = mean(self.rewards)
        stats['step_reward_median'] = median(self.rewards)
        stats['step_reward_max'] = max(self.rewards)
        stats['step_reward_min'] = min(self.rewards)
        stats['len'] = len(self.rewards)
        stats['entropy_mean'] = mean(entropy(self.mcts_policies))

        for k in self.infos[0].keys():
            vals = []
            for i in range(self.size()):
                vals.append(self.infos[i][k])

            if not (isinstance(vals[0], float) or isinstance(vals[0], int) or isinstance(vals[0], bool)):
                continue
            stats[f'{k}_sum'] = sum(vals)
            stats[f'{k}_mean'] = mean(vals)
            stats[f'{k}_median'] = median(vals)
            stats[f'{k}_max'] = max(vals)
            stats[f'{k}_min'] = min(vals)
        return stats

    @staticmethod
    def compute_stats_buffers(buffers: List['TransitionBuffer']):
        stats = {}
        for buffer in buffers:
            for k, v in buffer.get_stats().items():
                if k not in stats:
                    stats[k] = [v]
                else:
                    stats[k].append(v)
        return stats

    def update_priorities(self, batch_indices, new_priorities):
        for i, prio in enumerate(new_priorities):
            self.priorities[batch_indices[i]] = prio

    @staticmethod
    def log(summary_writer, update_step, log_dict, prefix='rollout'):
        for k, v in log_dict.items():
            summary_writer.add_scalar(f'{prefix}/{k}_mean', mean(v), update_step)
            summary_writer.add_scalar(f'{prefix}/{k}_median', median(v), update_step)
            summary_writer.add_scalar(f'{prefix}/{k}_min', min(v), update_step)
            summary_writer.add_scalar(f'{prefix}/{k}_max', max(v), update_step)


@ray.remote
class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.size: int = size
        self.transitions: TransitionBuffer = TransitionBuffer()

    def add(self, transitions: Union[TransitionBuffer, List[TransitionBuffer]]) -> None:
        if isinstance(transitions, list):
            transitions = TransitionBuffer.fuse_buffers(transitions)

        self.transitions.extend(transitions)
        num_drop = self.transitions.size() - self.size
        if num_drop > 0:
            self.transitions.drop_first_n(num_drop)

    def sample(self, batch_size, frame_stack) -> Tuple[TrainingBatch, np.ndarray]:
        return self.transitions.sample_training_batch(batch_size, frame_stack)

    def size(self) -> int:
        return len(self.transitions.obs)

    def clear(self) -> None:
        self.transitions = TransitionBuffer()

    def update_priorities(self, batch_indices, new_priorities):
        self.transitions.update_priorities(batch_indices, new_priorities)

