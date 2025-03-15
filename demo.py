# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import fusion

f = fusion.Fusion()
fg = fusion.FusionGuard(f)

tv0 = fusion.TensorViewBuilder().num_dims(3).shape([2, 4, 8]).contiguity(True).build()
tv1 = fusion.TensorViewBuilder().num_dims(3).shape([2, 4, 8]).contiguity(True).build()
f.add_input(tv0)
f.add_input(tv1)

tv2 = fusion.ops.add(tv0, tv1)
f.add_output(tv2)


print("Fusion IR")
f.print_math()

print("TensorView:")
print(tv0.to_string(0))
print(tv1.to_string(0))
print(tv2.to_string(0))
print("=========\n")

print("IterDomain:")
print(tv0.axis(0).to_string())
print(tv1.axis(0).to_string())
print(tv2.axis(0).to_string())
print("=========\n")

print("IterDomain Extent:")
print(tv0.axis(0).extent().to_string(0))
print(tv1.axis(0).extent().to_string(0))
print(tv2.axis(0).extent().to_string(0))
print("=========\n")

print("Fusion Executor Cache:")
fec = fusion.FusionExecutorCache(f)
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
