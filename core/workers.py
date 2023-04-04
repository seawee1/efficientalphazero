from copy import deepcopy
import time

import numpy as np
import ray

from core.mcts import BatchTree, MCTS
from config.base import BaseConfig
from core.replay_buffer import TransitionBuffer, ReplayBuffer, MCTSRollingWindow
from core.storage import SharedStorage
from core.util import MinMaxStats


class MCTSWorker:
    def __init__(
            self,
            config: BaseConfig,
            device: str,
            amp: bool,
            num_envs: int,
            use_dirichlet: bool,
    ):
        self.config = config
        self.model = self.config.create_model(device, amp)
        self.model.eval()
        self.num_envs = num_envs
        self.use_dirichlet = use_dirichlet

        self.envs = [config.env_creator() for _ in range(self.num_envs)]
        self.env_observation_space = self.envs[0].observation_space
        self.env_action_space = self.envs[0].action_space

    def collect(self):
        roots = BatchTree(self.num_envs, self.envs[0].action_space.n, self.config)  # Prepare datastructures
        mcts = MCTS(self.config, self.model)
        transition_buffers = [TransitionBuffer() for _ in range(self.num_envs)]
        mcts_windows = [MCTSRollingWindow(self.config.obs_shape, self.config.frame_stack) for _ in range(self.num_envs)]
        finished = [False] * self.num_envs

        for i, env in enumerate(self.envs):  # Initialize rolling windows for frame stacking
            mcts_windows[i].add(env.reset(), env.get_state())

        while not all(finished):
            # Prepare roots
            priors, values = self.model.compute_priors_and_values(mcts_windows)  # Compute priors and values for nodes to be expanded

            noises = None  # Inject noise into priors if configured
            if self.use_dirichlet:
                noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.env_action_space.n).astype(np.float32) for _ in range(self.num_envs)]
            roots.prepare(mcts_windows, self.config.root_exploration_fraction, priors, noises)

            windows = deepcopy(mcts_windows)
            root_visit_dists, root_values = mcts.search(roots, windows)  # Do MCTS search

            # Execute action sampled from MCTS policy
            actions = []
            for env_index, visit_dist in enumerate(root_visit_dists):
                if finished[env_index]:  # We can skip this, because this sub environment is done
                    actions.append(None)
                    continue

                # Calculate MCTS policy
                assert sum(visit_dist) > 0
                mcts_policy = visit_dist / np.sum(visit_dist)  # Convert child visit counts to probability distribution (TODO: temperature)

                # Take maximum visited child as action
                # We do it like this as to randomize action selection for case where visit counts are equal
                action = np.random.choice(np.argwhere(mcts_policy == np.max(mcts_policy)).flatten())
                #action = np.random.choice(range(self.env_action_space.n), p=mcts_policy)  # We could also sample instead of maxing
                actions.append(action)

                obs, reward, done, info = self.envs[env_index].step(action)  # Apply action

                # Priority by value error
                # priority = nn.L1Loss(reduction='none')(torch.Tensor([values[env_index]]), torch.Tensor([root_values[env_index]])).item()
                # priority += 1e-5
                # TODO: obs vs mcts_window obs
                transition_buffers[env_index].add_one(  # Add experience to data storage
                    mcts_windows[env_index].latest_obs(),  # The observation the action is based upon (vs. `obs`, which is the observation the action generated)
                    reward,
                    done,
                    info,
                    mcts_policy,
                    root_values[env_index],
                    mcts_windows[env_index].env_state,
                    1.0  # TODO
                )

                mcts_windows[env_index].add(obs, self.envs[env_index].get_state(), reward=reward, action=action, info=info)  # Update rolling window for frame stacking

                if done:
                    finished[env_index] = True
                    if not self.config.root_value_targets:  # Overwrite root values calculated during MCTS search with actual trajectory state returns
                        transition_buffers[env_index].augment_value_targets(max if self.config.max_reward_return else sum)

                    # Priority by "goodness"
                    # accu = max if self.config.max_reward_return else sum
                    # priorities = [accu(transition_buffers[env_index].rewards)] * transition_buffers[env_index].size()
                    # transition_buffers[env_index].priorities = priorities

            roots.apply_actions(actions)  # Move the tree roots to the new nodes of actions taken

        roots.clear()
        return transition_buffers


@ray.remote
class RolloutWorker(MCTSWorker):
    def __init__(
            self,
            config: BaseConfig,
            device: str,
            amp: bool,
            replay_buffer: ReplayBuffer,
            storage: SharedStorage
    ):
        num_envs = config.num_envs_per_worker
        use_dirichlet = config.use_dirichlet
        super().__init__(config, device, amp, num_envs, use_dirichlet)

        self.replay_buffer = replay_buffer
        self.storage = storage

    def run(self):

        while True:  # Wait for start signal
            if not ray.get(self.storage.get_start_signal.remote()):
                time.sleep(1)
                continue
            break

        collect_update_step = -1
        while True:
            # Check if training finished
            update_step = ray.get(self.storage.get_counter.remote())
            if update_step >= self.config.training_steps:
                time.sleep(30)
                break

            if collect_update_step == update_step:
                time.sleep(5)
                continue

            # Update weights
            model_weights = ray.get(self.storage.get_weights.remote())
            self.model.set_weights(model_weights)

            # Collect data
            transition_buffers = []
            while len(transition_buffers) < self.config.min_num_episodes_per_worker:
                transition_buffers.extend(self.collect())

            # Add episode data to replay buffer and stats to storage
            stats = TransitionBuffer.compute_stats_buffers(transition_buffers)
            self.storage.add_rollout_worker_logs.remote(stats)
            self.replay_buffer.add.remote(transition_buffers)

            collect_update_step = update_step
            self.storage.incr_workers_finished.remote()


@ray.remote
class TestWorker(MCTSWorker):
    def __init__(
            self,
            config: BaseConfig,
            device: str,
            amp: bool,
    ):
        num_envs = config.num_envs_per_worker
        use_dirichlet = config.test_use_dirichlet
        super().__init__(config, device, amp, num_envs, use_dirichlet)

        self.stats = None

    def run(self, model_weights, num_episodes):
        transition_buffers = []

        self.model.set_weights(model_weights)

        # Collect data
        while len(transition_buffers) < num_episodes:
            transition_buffers.extend(self.collect())

        # Compute and store stats
        self.stats = TransitionBuffer.compute_stats_buffers(transition_buffers)

    def get_stats(self):
        return self.stats


@ray.remote
class DemonstrationWorker:
    def __init__(
            self,
            config: BaseConfig,
            replay_buffer: ReplayBuffer,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.env = config.demonstrator_env_creator()

    def collect(self):
        transition_buffer = TransitionBuffer()
        demo_traj = self.config.collect_demonstration(self.env)

        for i in range(len(demo_traj['obs'])):
            obs = demo_traj['obs'][i]
            rew = demo_traj['rewards'][i]
            done = demo_traj['dones'][i]
            info = demo_traj['infos'][i]
            env_state = demo_traj['env_states'][i]
            priority = 1.0

            action = demo_traj['actions'][i]
            mcts_policy = np.zeros(self.env.action_space.n)
            mcts_policy[action] = 1.0

            transition_buffer.add_one(
                obs,
                rew,
                done,
                info,
                mcts_policy,
                None,  # Will be calculated afterwards
                env_state,
                priority
            )

        transition_buffer.augment_value_targets(max if self.config.max_reward_return else sum)  # Computes state returns based on rewards
        return transition_buffer

    def run(self):
        # Wait until start signal
        while ray.get(self.replay_buffer.size.remote()) < self.config.demo_buffer_size:
            transition_buffer = self.collect()
            self.replay_buffer.add.remote(transition_buffer)

