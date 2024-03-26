import torch


def get_spike_fn(name, *args):
    """
    Returns a spike function that can be called, and allows to pass arguments for e.g. the shape of the surrogate gradient.
    """
    args = [torch.tensor(arg) for arg in args]  # easier to have them as tensors

    def inner(x):
        return eval(name).apply(x, *args)  # call spike fn with args

    return inner


class BaseSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *args):
        ctx.save_for_backward(x, *args)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class FastSigmoid(BaseSpike):
    """
    Spike function with derivative of fast sigmoid as surrogate gradient. From Zenke et al., Neural Computation 2018.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = height / (1 + slope * x.abs()) ** 2
        return grad_input * sg, None, None


class ArcTan(BaseSpike):
    """
    Spike function with derivative of arctan as surrogate gradient. From Fang et al., arXiv 2020.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = height / (1 + slope * x * x)
        # print((1 + slope * x * x).min(), grad_input.isnan().any(), grad_input.isfinite().all())
        # if not grad_input.isfinite().all():
        #     print(grad_input)
        return grad_input * sg, None, None
