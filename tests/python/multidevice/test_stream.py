# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from nvfuser_direct import FusionDefinition


def test_allgather_matmul(multidevice_direct_test):
    with FusionDefinition() as fd:
        x = fd.define_tensor([])
