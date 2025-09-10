# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import re
from nvfuser_direct import FusionDefinition, DataType  # noqa: F401


def find_first_mismatch(str1, str2):
    try:
        return next(i for i, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2)
    except StopIteration:
        return -1 if len(str1) == len(str2) else min(len(str1), len(str2))


def execute_and_check(test_str):
    exec(test_str)
    with FusionDefinition() as fd:
        eval("nvfuser_fusion")(fd)

    fd_str = fd.__str__()
    fd_str = re.sub("nvfuser_fusion", "nvfuser_fusion", fd_str)
    index = find_first_mismatch(fd_str, test_str)
    assert index == -1, f"Mismatch at: {fd.__str__()[index:]}"


def test_broadcast_in_dim_inline_defs():
    constant_bcast_in_dim_example = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[False, False], dtype=DataType.Float, is_cpu=False)
    tv2 = fd.ops.broadcast_in_dim(tv0, shape=[2, 2, 2], broadcast_dims=[0, 1])
    fd.add_output(tv2)
"""
    execute_and_check(constant_bcast_in_dim_example)

    mixed_bcast_in_dim_example = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[False, False], dtype=DataType.Float, is_cpu=False)
    c3 = fd.define_scalar(None, dtype=DataType.Index)
    tv2 = fd.ops.broadcast_in_dim(tv0, shape=[2, 2, c3], broadcast_dims=[0, 1])
    fd.add_output(tv2)
"""
    execute_and_check(mixed_bcast_in_dim_example)


def test_inline_def_reshape():
    constant_reshape_example = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1], contiguity=[False], dtype=DataType.Float, is_cpu=False)
    tv1 = fd.ops.reshape(tv0, new_shape=[2, 5])
    fd.add_output(tv1)
"""
    execute_and_check(constant_reshape_example)

    mixed_reshape_example = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1], contiguity=[False], dtype=DataType.Float, is_cpu=False)
    c2 = fd.define_scalar(None, dtype=DataType.Index)
    tv1 = fd.ops.reshape(tv0, new_shape=[2, c2])
    fd.add_output(tv1)
"""
    execute_and_check(mixed_reshape_example)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
