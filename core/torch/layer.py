import torch
import torch.nn as nn

from core.torch.neuron import *
from core.torch.synapse import *


__all__ = ["Rnn", "Conv3dRelu", "Conv3dTanh", "Conv3dIdentity", "Conv3dCubaLif", "TraceConv3dCubaLif", "LinearRelu", "LinearTanh", "LinearIdentity", "LinearCubaLif", "LinearCubaSoftLif", "Conv3dRnn", "Conv3dRnnCubaLif", "LinearRnnCubaLif"]


class Conv3dRelu(nn.Module):
    """
    Layer without explicit recurrency (no recurrent connections), but can have implicit recurrency if the modules are set as such.

    Furthermore, it can be made spiking or non-spiking, depending on the neuron module.

    Note that both synapse and neuron state are tracked, so both can be stateful.
    """

    synapse_module = Conv3d
    neuron_module = Relu

    def __init__(self, synapse, neuron):
        super().__init__()

        self.synapse = self.synapse_module(**synapse)
        self.neuron = self.neuron_module(**neuron)

        self.state = [None, None]  # synapse, neuron

    def forward(self, input):
        synapse_state, neuron_state = self.state

        x, synapse_state = self.synapse(input, synapse_state)
        x, neuron_state = self.neuron(x, neuron_state)

        self.state = [synapse_state, neuron_state]
        return x

    def reset(self):
        self.state = [None, None]


class Conv3dTanh(Conv3dRelu):
    neuron_module = Tanh


class Conv3dIdentity(Conv3dRelu):
    neuron_module = Identity


class Conv3dCubaLif(Conv3dRelu):
    neuron_module = CubaLif


class TraceConv3dCubaLif(Conv3dCubaLif):
    synapse_module = TraceConv3d


class LinearRelu(Conv3dRelu):
    synapse_module = Linear


class LinearTanh(Conv3dRelu):
    synapse_module = Linear
    neuron_module = Tanh


class LinearIdentity(Conv3dIdentity):
    synapse_module = Linear


class LinearCubaLif(Conv3dRelu):
    synapse_module = Linear
    neuron_module = CubaLif

class LinearCubaSoftLif(LinearCubaLif):
    neuron_module = CubaSoftLif

class Rnn(nn.Module):
    """
    Layer with explicit recurrency (recurrent connections).

    By setting the class attributes, we can make all kinds of variations, as you can see below!
    """

    synapse_ff_module = Linear
    synapse_rec_module = Linear
    neuron_module = Tanh

    def __init__(self, synapse_ff, synapse_rec, neuron):
        super().__init__()

        self.synapse_ff = self.synapse_ff_module(**synapse_ff)
        self.synapse_rec = self.synapse_rec_module(**synapse_rec)
        self.neuron = self.neuron_module(**neuron)

        self.state = [None, None, None]  # synapse ff, synapse rec, neuron

    def forward(self, input):
        synapse_ff_state, synapse_rec_state, neuron_state = self.state

        x_ff, synapse_ff_state = self.synapse_ff(input, synapse_ff_state)
        x = neuron_state[-1] if neuron_state is not None else torch.zeros_like(x_ff)  # assumes last is previous output
        x_rec, synapse_rec_state = self.synapse_rec(x, synapse_rec_state)
        x, neuron_state = self.neuron(x_ff + x_rec, neuron_state)

        self.state = [synapse_ff_state, synapse_rec_state, neuron_state]
        return x

    def reset(self):
        self.state = [None, None, None]

class LinearRnnCubaLif(Rnn):
    synapse_ff_module = Linear
    synapse_rec_module = Linear
    neuron_module = CubaLif

class Conv3dRnn(Rnn):
    synapse_ff_module = Conv3d
    synapse_rec_module = Conv3d


class Conv3dRnnCubaLif(Conv3dRnn):
    neuron_module = CubaLif

