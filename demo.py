# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from nvfuser import fusion

f = fusion.Fusion()
fg = fusion.FusionGuard(f)

tv0 = fusion.TensorViewBuilder().n_dims(1).shape([10]).contiguity(True).build()
tv1 = fusion.TensorViewBuilder().n_dims(1).shape([10]).contiguity(True).build()
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

# TODO Fix Segmentation Fault atexit of python script
