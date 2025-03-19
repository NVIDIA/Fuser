# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import direct
from direct_fusion_definition import FusionDefinition

fd = FusionDefinition()
tv0 = direct.TensorViewBuilder().num_dims(3).shape([2, 4, 8]).contiguity(True).build()
tv1 = direct.TensorViewBuilder().num_dims(3).shape([2, 4, 8]).contiguity(True).build()

fd.add_input(tv0)
fd.add_input(tv1)
tv2 = fd.ops.add(tv0, tv1)
fd.add_output(tv2)


print("Fusion IR")
fd.fusion.print_math()

print("TensorView:")
print(tv0)
print(tv1)
print(tv2)
print("=========\n")

print("TensorDomain")
print(tv0.domain())
print(tv1.domain())
print(tv2.domain())
print("=========\n")

print("IterDomain:")
print(tv0.axis(0))
print(tv1.axis(0))
print(tv2.axis(0))
print("=========\n")

print("IterDomain Extent:")
print(tv0.axis(0).extent())
print(tv1.axis(0).extent())
print(tv2.axis(0).extent())
print("=========\n")

print("Fusion Executor Cache:")
fec = direct.FusionExecutorCache(fd.fusion)
inputs = [
    torch.ones(2, 4, 8, device="cuda"),
    torch.ones(2, 4, 8, device="cuda"),
]
print("---")
print(fec.execute(inputs))
print("---")
print(fec.is_compiled(inputs))
print("---")
fec.fusion().print_math()
print("---")
print(fec.print_fusion())
print("---")
print(fec.get_scheduled_ir(inputs))
print("---")
print(fec.get_cuda_kernel(inputs))
print("---")
print(fec.get_most_recent_scheduled_ir())
print("=========\n")
