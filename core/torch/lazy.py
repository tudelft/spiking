import torch
import torch.nn as nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter


class BaseModuleMixin(LazyModuleMixin, nn.Module):
    """
    Mixin to make custom lazy PyTorch modules that can have learnable and non-learnable parameters.

    Dynamics is a dict where the keys are the names of the parameters and the values determine the shape and distribution of the parameters:
    - If the value is a list/tuple, every channel/output feature will have its own parameter, and the elements in the list set the mean and std of the normal init
    - If the value is a number, all channels/output features will share the same parameter, and the number sets the init

    Learnable is a list of the parameter names that are learnable, while the rest are fixed.
    """

    def __init__(self, dynamics, learnable, *args, **kwargs):
        super().__init__(*args, **kwargs)  # needed for mixin

        # go over all parameters
        for name in dynamics.keys():
            # just like in normal torch, parameters track gradients, buffers don't
            # buffers are nice because they automatically get saved in state dict and move to GPU with model
            if name in learnable:
                self.register_parameter(name, None)  # make sure it's in state dict
                setattr(self, name, UninitializedParameter())  # no shape yet
            else:
                self.register_buffer(name, None)  # make sure it's in state dict
                setattr(self, name, UninitializedBuffer())  # no shape yet

        self.dynamics = dynamics
        self.learnable = learnable

    def initialize_parameters(self, input, *_):  # catches and discards state argument
        """
        This overrides the LazyModuleMixin method to initialize the parameters of the module.
        """
        # only init if not done yet
        if self.has_uninitialized_params():
            with torch.no_grad():
                # if input (b, o), then linear connection, and ks = 0
                # if input (b, c, h, w), then conv2d connection, and ks = 2
                # if input (b, c, d, h, w), then conv3d connection, and ks = 3
                c = input.shape[1]  # channels (c) or output features (o)
                ks = input.ndim - 2  # kernel dimensions (if any)
                for name, value in self.dynamics.items():
                    param = getattr(self, name)
                    if isinstance(value, (list, tuple)):  # per channel or output feature
                        param.materialize((c,) + (1,) * ks)
                        param.normal_(value[0], value[1])  # normal distribution for now
                    else:  # per layer
                        param.materialize((1,) + (1,) * ks)
                        param.fill_(value)  # constant value
