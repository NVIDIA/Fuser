from collections import namedtuple
from make_tensor import make_tensor, all_dtypes
from functools import partial, wraps
from typing import Callable
import torch

Domain = namedtuple("Domain", "low high")


class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "args",
        "kwargs",
    ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"[SampleInput args={self.args} kwargs={self.kwargs}]"


# TODO: require use of generic Thunder dtypes (once they exist)
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op: Callable,
        name: str,
        *,
        dtypes=None,
        sample_input_generator,
        error_input_generator=None,
        torch_reference=None,
        numpy_reference=None,
        jax_reference=None,
        domain=(None, None),
    ):
        self.op = op
        self.name = name
        self._dtypes = dtypes if dtypes is not None else all_dtypes
        self.sample_input_generator = sample_input_generator
        self.error_input_generator = error_input_generator
        self.torch_reference = torch_reference
        self.numpy_reference = numpy_reference
        self.jax_reference = jax_reference
        self.domain = Domain(*domain)

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def sample_inputs(
        self, torch_dtype: torch.dtype, *, requires_grad: bool = False, **kwargs
    ):
        return self.sample_input_generator(self, torch_dtype, requires_grad, **kwargs)

    def error_inputs(self, **kwargs):
        return self.error_input_generator(self, **kwargs)

    def dtypes(self):
        return self._dtypes


# TODO Add small value, large value, and extremal-valued samples
def elementwise_unary_generator(
    op,
    dtype: torch.dtype,
    requires_grad: bool,
    *,
    supports_numbers: bool = True,
    **kwargs,
):
    low = None if op.domain.low is None else max(-9, op.domain.low)
    high = None if op.domain.high is None else min(9, op.domain.high)
    make_arg = partial(
        make_tensor,
        device="cuda",
        dtype=dtype,
        low=low,
        high=high,
        requires_grad=requires_grad,
        **kwargs,
    )

    shapes = (
        # TODO: restore size zero cases
        # (0, 2, 1),
        # (5, 0, 3),
        # (),
        (11,),
        (4, 4),
        (1024, 1024),
        (64, 64, 64),
    )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape))

    # Noncontiguous inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape, noncontiguous=True))

    # Arbitrarily strided inputs
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        yield SampleInput(a)


def _elementwise_unary_torch(op):
    @wraps(op)
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return op(x)

        return op(torch.tensor(x)).item()

    return _fn
