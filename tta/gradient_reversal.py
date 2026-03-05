"""Gradient Reversal Layer (GRL) for Domain-Adversarial Training.

During the forward pass the input is returned unchanged.
During the backward pass the gradient is negated and scaled by ``lambda_``.
"""
from torch import Tensor
from torch.autograd import Function


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: Tensor, lambda_: float = 1.0) -> Tensor:
    """Apply gradient reversal to *x* with scaling factor *lambda_*."""
    return _GradientReversalFn.apply(x, lambda_)