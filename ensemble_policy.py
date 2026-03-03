"""
Ensemble rollout policies for data collection and evaluation.

Policies:
- EnsembleTransformerPolicy: Averages softmax outputs from multiple transformers
"""

import numpy as np
import scipy.special
import torch

from get_rollout_policy import TransformerPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsembleTransformerPolicy(TransformerPolicy):
    """
    Policy that averages softmax outputs from an ensemble of Decision Transformers.
    Shares context buffers across all members (they see the same history).
    """

    def __init__(self, models, temp=0.1, context_horizon=None, env_horizon=None,
                 sliding_window=True):
        super().__init__(models[0], temp, context_horizon, env_horizon, sliding_window)
        self.models = models

    @torch.no_grad()
    def get_action(self, states):
        for m in self.models:
            m.eval()

        current_states = torch.from_numpy(states).float().to(device)

        if len(self.context_states) < 1:
            all_logits = [
                m.get_action(current_states, None, None, None, None)
                for m in self.models
            ]
        else:
            ctx_states, ctx_actions, ctx_rewards, ctx_dones = self._get_context_tensors()
            all_logits = [
                m.get_action(current_states, ctx_states, ctx_actions, ctx_rewards, ctx_dones)
                for m in self.models
            ]

        all_probs = np.stack([
            scipy.special.softmax(logits.cpu().numpy() / self.temp, axis=1)
            for logits in all_logits
        ])
        mean_probs = all_probs.mean(axis=0)

        batch_size, num_actions = mean_probs.shape
        action_ids = np.array([
            np.random.choice(num_actions, p=mean_probs[i])
            for i in range(batch_size)
        ])
        actions = np.zeros((batch_size, num_actions))
        actions[np.arange(batch_size), action_ids] = 1.0
        return actions


