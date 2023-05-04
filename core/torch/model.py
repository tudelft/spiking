import torch.nn as nn

from layer import *


def get_model(model, config, data=None, device=None):
    """
    We use lazy PyTorch modules, so we need to initialize the shapes by tracing them with a batch of data.
    """
    # load model from config
    if isinstance(model, str):
        model = eval(model)(**config)
    else:
        model = model(**config)

    # trace model with a batch of data
    if data is not None:
        model.trace(data)

    # move to device
    model.to(device)
    return model


class BaseModel(nn.Module):
    """
    Base model with shared methods (tracing).
    """

    def __init__(self):
        super().__init__()

        # tracing (easier than checking for uninitialized params)
        self.traced = False

    def forward(self, input):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def trace(self, x):
        """
        Trace the model (forward a batch of data through it) to initialize the shapes of the parameters.
        """
        # only if not traced yet
        if not self.traced:
            self.traced = True
            self.forward(x.cpu())  # assumes model hasn't been moved to GPU yet
            self.reset()


class FullyConvCubaLif(BaseModel):
    """
    Example model, fully convolutional, with CUBA LIF neurons.

    Class attributes allow to make different models by only changing these.
    """

    encoder_layer = Conv3dCubaLif
    prediction_layer = Conv3dIdentity

    def __init__(self, e1, e2, p1):  # all config dicts for the respective layers
        super().__init__()

        # layers
        self.e1 = self.encoder_layer(**e1)
        self.e2 = self.encoder_layer(**e2)
        self.p1 = self.prediction_layer(**p1)

    def forward(self, input):
        x = self.e1(input)
        x = self.e2(x)
        x = self.p1(x)
        return x

    def reset(self):
        self.e1.reset()
        self.e2.reset()
        self.p1.reset()
