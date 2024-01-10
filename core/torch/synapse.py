from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.torch.lazy import BaseModuleMixin


__all__ = ["Linear", "Conv3d", "TraceConv3d"]


class Linear(nn.LazyLinear):
    """
    Simple wrapper to make synapse return state.
    """

    cls_to_become = None  # prevent from becoming nn.Linear

    def forward(self, input, state):
        input = super().forward(input)
        return input, [input]
    
    def initialize_parameters(self, input, *_):  # catch and discard state argument
        super().initialize_parameters(input)


class Conv3d(nn.LazyConv3d):
    """
    Convolution with a 3D kernel (d, h, w) allows to have synapses with multiple delays between any two neurons, meaning they look at the last d steps of input. If you want regular 2D convolution, just set d to 1.

    We make a buffer of inputs with deque.
    """

    cls_to_become = None  # prevent from becoming nn.Conv3d

    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input, state):
        input = input.unsqueeze(2)  # add delay dimension; assumes network receives single timesteps
        state = self.reset_state(input) if state is None else state
        (input_buffer,) = state

        # add incoming to buffer, then cat to be fed to conv3d
        input_buffer.append(input)
        input = torch.cat(list(input_buffer), dim=2)

        x = super().forward(input)

        return x.squeeze(2), [input_buffer]  # remove delay dimension again for neuron

    def reset_state(self, input):
        _, _, buffer_size, _, _ = self.weight.shape  # kernel_size is (d, h, w), so buffer_size is d
        return [deque([torch.zeros_like(input) for _ in range(buffer_size)], maxlen=buffer_size)]

    def initialize_parameters(self, input, *_):  # catch and discard state argument
        super().initialize_parameters(input.unsqueeze(2))  # add delay dimension


class TraceConv3d(BaseModuleMixin, nn.LazyConv3d):
    """
    Compared to Conv3d, this one maintains a trace/moving average of the input, whose max is subtracted. This serves as a homeostatic mechanism, as explained in Paredes-Valles et al., TPAMI 2019.

    Inheritance is nice because it allows stateless (nn.LazyConvXd) and stateful (this and Conv3d) synapses to be on the same level. Only needs an init fn if you want to pass extra arguments; else mixin already passes on args.
    """

    cls_to_become = None  # prevent from becoming nn.Conv3d

    def __init__(
        self, dynamics, learnable, max_delay, out_channels, kernel_size, stride=1, padding=0, bias=False, w_init=None
    ):
        # weight init
        # needs to be set before reset_parameters is called
        self.w_init = w_init

        # first two args are for mixin
        super().__init__(dynamics, learnable, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        # delay: we have a buffer of size max_delay
        # and from that we select kernel_size[0] values with indices delay_steps
        self.max_delay = max_delay
        self.delay_steps = torch.linspace(0, max_delay - 1, kernel_size[0]).long()

    def forward(self, input, state):
        input = input.unsqueeze(2)  # add delay dimension; assumes network receives single timesteps
        state = self.reset_state(input) if state is None else state
        input_buffer, x = state

        # update buffer
        # if max_delay is 3 with 3 delays, forcing function will consider delays 0-2
        input_buffer.append(input)
        input = torch.cat(list(input_buffer), dim=2)  # cat because inputs are 5D
        input = input[:, :, self.delay_steps]  # select only delay steps

        # presynaptic trace
        # not an exact first-order system because add_x != (1 - leak_x)
        leak_x = torch.sigmoid(self.leak_x)
        add_x = torch.sigmoid(self.add_x)
        x = x * leak_x + add_x * input

        # max trace
        # should only consider max across direct spatial neighborhood (size 3x3 for now)
        x_max = F.conv3d(x, torch.ones_like(self.weight), stride=self.stride, padding=self.padding)  # make into patches
        x_max = F.max_pool2d(x_max.squeeze(2), 3, stride=1, padding=1)  # here's the 3

        # current
        i = super().forward(input)  # works because BaseModuleMixin doesn't have forward
        i = i - x_max.unsqueeze(2)  # removed delay dim for max_pool2d, now add it back

        return i.squeeze(2), [input_buffer, x]  # remove delay dimension again for neuron

    def reset_state(self, input):
        """
        This synapse can have a spacing in the delayed inputs it selects, meaning we select d values from a buffer of size max_delay.
        """
        _, _, d, _, _ = self.weight.shape  # number of values to select based on kernel size
        input_buffer = deque(
            [torch.zeros_like(input) for _ in range(self.max_delay)], maxlen=self.max_delay
        )  # buffer for all steps
        b, c, _, h, w = input.shape
        x = torch.zeros(b, c, d, h, w, dtype=input.dtype, device=input.device)  # trace same size as selected inputs
        return [input_buffer, x]

    def reset_parameters(self):
        """
        Optionally overwrites the reset_parameters method of nn.Conv3d if we give an init distribution for the weights.
        """
        if self.w_init is None:
            super().reset_parameters()  # original nn.Conv3d
        else:
            if not self.has_uninitialized_params() and self.in_channels != 0:  # check from original nn.Conv3d
                assert self.bias is None  # only doing weight init here
                nn.init.uniform_(self.weight, -self.w_init, self.w_init)  # uniform init for now

    def initialize_parameters(self, input, *_):  # catch and discard state argument
        input = input.unsqueeze(2)  # add delay dimension
        super().initialize_parameters(input)  # calls BaseModuleMixin
        nn.LazyConv3d.initialize_parameters(self, input)  # calls nn.LazyConv3d
