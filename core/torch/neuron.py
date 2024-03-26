import torch
import torch.nn as nn

from core.torch.lazy import BaseModuleMixin
from core.torch.utils import get_spike_fn


__all__ = ["Relu", "Tanh", "Identity", "CubaLif", "CubaSoftLif", "Li"]



def print_grad(grad):
    # print(grad.isfinite().all())
    # if not grad.isfinite().all():
    #     print(grad)
    print(grad.abs().max())
    return grad

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

class Li(BaseModuleMixin):
    """
    Current-based LIF model with optional learnable parameters.

    The mixin allows us to use the same code for the learnable and non-learnable versions.

    Leaks are clamped to [0, 1] by sigmoids and the threshold is clamped to [0, inf] by ReLU.

    Voltage reset is hard, and we decouple the leak from the increment for both current and voltage updates (so they don't represent first-order systems exactly).
    """

    def forward(self, input, state):
        state = self.reset_state(input) if state is None else state
        i, v = state

        # current update
        leak_i = torch.sigmoid(self.leak_i)
        i = i * leak_i + input

        # voltage update with hard reset
        leak_v = torch.sigmoid(self.leak_v)
        v = v * leak_v + i

        return v, torch.stack([i, v])

    def reset_state(self, input):
        return torch.zeros(2, *input.shape, dtype=input.dtype, device=input.device)


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

    @staticmethod
    def voltage_update(v, leak_v, s, i):
        v = v * leak_v * (1 - s) + i
        return v

    def forward(self, input, state):
        state = self.reset_state(input) if state is None else state
        i, v, s = state

        # current update
        leak_i = torch.sigmoid(self.leak_i)
        # h_leak_i = leak_i.register_hook(print_grad)
        i = i * leak_i + input
        # print(i.abs().max())
        # h_i = i.register_hook(print_grad)

        # voltage update with hard reset
        leak_v = torch.sigmoid(self.leak_v)
        # h_leak_v = leak_v.register_hook(print_grad)
        v = v * leak_v * (1 - s) + i
        # print(v.max())
        # h_v = v.register_hook(print_grad)

        # spike!
        thresh = torch.relu(self.thresh)
        s = self.spike(v - thresh)
        # print(s.sum())

        return s, torch.stack([i, v, s])  # make sure that neuron output (spikes) are last

    def reset_state(self, input):
        return torch.zeros(3, *input.shape, dtype=input.dtype, device=input.device)



class CubaSoftLif(CubaLif):
    '''
    Soft reset for activation
    '''
    def forward(self, input, state):
        state = self.reset_state(input) if state is None else state
        i, v, s = state

        # current update
        leak_i = torch.sigmoid(self.leak_i)
        i = i * leak_i + input
        # h_i = i.register_hook(print_grad)
        # i = torch.relu(i)
        # print(i.abs().max())
    

        # voltage update with hard reset
        leak_v = torch.sigmoid(self.leak_v)
        thresh = torch.relu(self.thresh)

        v = (v * leak_v) - ((1 - s) * thresh) + i
        # v = torch.relu(v)
        # leak_v_i = leak_v.register_hook(print_grad)
        # v = 3 * torch.sigmoid(v)
        # print(v.max())

        # spike!
        s = self.spike(v - thresh)
        # print(s.sum())
        # print(i.isnan().any(), v.isnan().any(), s.isnan().any(), input.isnan().any(), i.isfinite().all(), v.isfinite().all(), s.isfinite().all(), input.isfinite().all())

        return s, torch.stack([i, v, s])  # make sure that neuron output (spikes) are last