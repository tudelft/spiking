import torch
import torch.nn as nn

from lazy import BaseModuleMixin
from utils import get_spike_fn


__all__ = ["Relu", "Tanh", "Identity", "CubaLif"]


class Relu(nn.ReLU):
    """
    Simple wrapper to make the neuron return state.
    """

    def forward(self, input, state):
        x = super().forward(input)
        return x, [x]


class Tanh(nn.Tanh):
    def forward(self, input, state):
        x = super().forward(input)
        return x, [x]


class Identity(nn.Module):
    def forward(self, input, state):
        return input, [input]


class CubaLif(BaseModuleMixin):
    """
    Current-based LIF model with optional learnable parameters.

    The mixin allows us to use the same code for the learnable and non-learnable versions.

    Leaks are clamped to [0, 1] by sigmoids and the threshold is clamped to [0, inf] by ReLU.

    Voltage reset is hard, and we decouple the leak from the increment for both current and voltage updates (so they don't represent first-order systems exactly).
    """

    def __init__(self, dynamics, learnable, spike_fn):
        super().__init__(dynamics, learnable)
        self.spike = get_spike_fn(spike_fn["name"], *spike_fn["shape"])

    def forward(self, input, state):
        state = self.reset_state(input) if state is None else state
        i, v, s = state

        # current update
        leak_i = torch.sigmoid(self.leak_i)
        i = i * leak_i + input

        # voltage update with hard reset
        leak_v = torch.sigmoid(self.leak_v)
        v = v * leak_v * (1 - s) + i

        # spike!
        thresh = torch.relu(self.thresh)
        s = self.spike(v - thresh)

        return s, torch.stack([i, v, s])  # make sure that neuron output (spikes) are last

    def reset_state(self, input):
        return torch.zeros(3, *input.shape, dtype=input.dtype, device=input.device)
