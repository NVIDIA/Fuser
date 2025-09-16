# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser_direct import FusionDefinition

verbose_ = False


def test_tutorial_memcpy():
    # First, we define a fusion. A common pattern is:
    # - Declare a Fusion, which works as a container of expressions using
    #   with context manager.
    # - Setup inputs. fd.define_tensor can be used to manually create tensors.
    #   fd.from_pytorch will create a TensorView given a pytorch tensor. Fusion
    #   registration is automatic.
    # - Define operations with the registered inputs.
    #   For supported operations, run:
    #   >>> import nvfuser_direct
    #   >>> fd = nvfuser_direct.FusionDefinition()
    #   >>> help(fd.ops)
    # - Most of operations that take tensors as inputs produce tensors
    #   as outputs, which can then be used as inputs to another
    #   operations.
    # - Final outputs should be set as fusion outputs with fd.add_output

    with FusionDefinition() as fd:
        # Create a 2D tensor of type float. It's "symbolic" as we do not
        # assume any specific shape except for that it's 2D.
        tv0 = fd.define_tensor(shape=[-1, -1])

        # Just create a copy
        tv1 = fd.ops.set(tv0)
        fd.add_output(tv1)

    if verbose_:
        # Here's some common ways to inspect the fusion. These are not
        # necessary for running the fusion but should provide helpful
        # information for understanding how fusions are transformed.

        # Print a concise representation of the fusion exprssions
        print(fd.fusion.print_math())

        # Generate and print a CUDA kernel. Notice that at this point the
        # genereated code is just a sequential kernel as we have not
        # scheduled the fusion yet, but it should be a valid CUDA kernel
        print(fd.fusion.print_kernel())

    # Next, try running the fusion. First, we need to set up a sample
    # input tensor. Here, we create a 32x32 tensor initialized with
    # random float values.

    t0 = torch.randn(32, 32, dtype=torch.float, device="cuda:0")

    # Next, lower the Fusion to Kernel, generate CUDA kernel source and then
    # compile it with nvrtc. After compilation, KernelExecutor now has a
    # compiled kernel, which can be executed as:
    outputs = fd.manual_execute([t0])

    # Note that this run is done using just one thread, which will be
    # corrected below.

    # To validate the output, we can just assert that the output is
    # equal to the input as this is just a copy fusion. More commonly,
    # though, fd.validate is used to validate outputs while
    # automatically adjusting thresholds of valid deviations.
    assert outputs[0].equal(t0)
