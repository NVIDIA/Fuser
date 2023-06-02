import pytest
import numpy as np
import math
import torch
from torch.testing import assert_close
from pytest_framework import ops, run_snippet
from pytest_opinfos import opinfos

from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.jit_utils import RUN_CUDA


# Will only create the nvfuser module if CUDA is available
try:
    from nvfuser import (
        FusionCache,
        FusionDefinition,
        DataType,
        Tensor,
        version,
        compute_contiguity,
    )
    from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
except ImportError:
    pass

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def unary_fusion_func(fd: FusionDefinition, operation, inputs):
    t0 = fd.from_pytorch(inputs[0])
    t1 = operation(fd)(t0)
    fd.add_output(t1)


def snippet_errors(nvf_op, sample, ex_type):
    ex = None
    try:
        with FusionDefinition() as fd:
            unary_fusion_func(fd, nvf_op, sample.args)
        fd.execute(*sample.args, **sample.kwargs)
    except Exception as e:
        ex = e

    assert ex is not None, f"Expected an exception"
    assert ex_type is type(
        ex
    ), f"Expected an exception with type {ex_type}, but found ex={ex}"


def snippet_torch_consistency(nvf_op, torch_op, sample):
    with FusionDefinition() as fd:
        unary_fusion_func(fd, nvf_op, sample.args)
    nvfuser_result = fd.execute(sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)

    if isinstance(nvfuser_result, Exception):
        raise nvfuser_result

    assert_close(nvfuser_result[0], torch_result, equal_nan=True, atol=1e-3, rtol=0)


@ops(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op):
    for sample, ex_type in op.error_inputs(device):
        result = run_snippet(snippet_errors, op, None, op.op, sample, ex_type)
        if result is not None:
            return result


@ops(tuple(op for op in opinfos if op.torch_reference is not None))
def test_consistency(op, dtype: torch.dtype):
    for sample in op.sample_inputs(dtype):
        result = run_snippet(
            snippet_torch_consistency,
            op,
            dtype,
            op.op,
            op.torch_reference,
            sample,
        )
        if result is not None:
            return result
