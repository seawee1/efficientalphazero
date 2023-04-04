from typing import List

import torch
import torch.nn as nn
import numpy as np
import ray

from config.base import BaseConfig
from core.replay_buffer import MCTSRollingWindow, TrainingBatch


class BaseModel(nn.Module):
    def __init__(self, config, obs_shape, num_act, device, amp):
        super().__init__()
        self.config: BaseConfig = config
        self.obs_shape = obs_shape
        self.num_act = num_act
        self.device = device
        self.amp = amp

    def forward(self):
        raise NotImplementedError

    def compute_priors_and_values(self):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


class StandardModel(BaseModel):
    def __init__(self, config, obs_shape, num_act, device, amp):
        super().__init__(config, obs_shape, num_act, device, amp)
        self.shared = None
        self.actor = None
        self.critic = None

    def forward(self, x):
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.amp):
            feat = self.shared(x)
            p = self.actor(feat)
            v = self.critic(feat)
        return p, v

    def compute_priors_and_values(self, windows: List[MCTSRollingWindow]):
        obs = np.stack([window.obs for window in windows])
        obs = torch.from_numpy(obs).to(self.device).float()

        with torch.no_grad():
            policy_logits, values_logits = self.forward(obs)

        priors = nn.Softmax(dim=-1)(policy_logits)
        values_softmax = nn.Softmax(dim=-1)(values_logits)
        values = self.config.phi_inverse_transform(values_softmax).flatten()

        if self.config.value_transform:
            values = self.config.inverse_scalar_transform(values)

        priors = priors.cpu().float().numpy()
        values = values.cpu().float().numpy()
        return priors, values

    def update_weights(self, train_batch, optimizer, scaler, scheduler):
        self.train()
        train_batch.to_torch(self.device)

        policy_logits, value_logits = self.forward(train_batch.obs)

        value_targets = train_batch.value_targets
        if self.config.value_transform:
            value_targets = self.config.scalar_transform(value_targets)
        value_targets_phi = self.config.phi_transform(value_targets)

        policy_loss = -(torch.log_softmax(policy_logits, dim=1) * train_batch.mcts_policies).mean()
        value_loss = -(torch.log_softmax(value_logits, dim=1) * value_targets_phi).mean()

        # Update prios
        """
        values_pred = self.config.phi_inverse_transform(value_logits)
        with torch.no_grad():
            new_priorities = nn.L1Loss(reduction='none')(values_pred, value_targets).cpu().numpy().flatten()
        new_priorities += 1e-5
        replay_buffer.update_priorities.remote(batch_indices, new_priorities)
        """
        parameters = self.parameters()

        total_loss = (policy_loss + value_loss) / 2
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        return total_loss, policy_loss, value_loss
