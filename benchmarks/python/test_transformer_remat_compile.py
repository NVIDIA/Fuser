# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import clear_cuda_cache
from .core import run_benchmark
import torch
from functools import partial

"""
This fusion is a benchmark for Issue: https://github.com/NVIDIA/Fuser/issues/2916.
To see the compile time behavior reported, please apply the following patch to Thunder
```
diff --git a/thunder/core/rematerialization.py b/thunder/core/rematerialization.py
index 7db7b88f..4967f0c9 100644
--- a/thunder/core/rematerialization.py
+++ b/thunder/core/rematerialization.py
@@ -347,7 +347,7 @@ def find_cut(
     def add_edges(var):
         var_name = var.name
         weight = get_weight(var)
-        weight = weight / 2.0 if var_name in (x.name for x in producer.args) else weight
+        weight = weight * 0.0 if var_name in (x.name for x in producer.args) else weight
         add_edge(var_name + "_in", var_name + "_out", capacity=weight)
         for user in combined_consumers._dict.get(var_name, tuple()):
             if user.sym.id in sym_skip_list:
```

Note, that, we use nvfuser matmul op to obtain a single fusion for the repro since the current benchmarking infra only measures host times for a single fusion definition using nvfuser fusion profiler.
"""


def remat_transformer_fusion(fd : FusionDefinition, num_layers: int) -> None :
    x = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
    weight = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
    
    x_add_mlp = x
    for _ in range(num_layers):
        x_normed_1 = fd.ops.tanh(x_add_mlp)
        attention_output = fd.ops.matmul(weight, x_normed_1)
        x_add_attn = fd.ops.add(attention_output, x_add_mlp)
        x_normed_2 = fd.ops.tanh(x_add_attn)
        mlp_output = fd.ops.matmul(weight, x_normed_2)
        x_add_mlp = fd.ops.add(mlp_output, x_add_attn)
    fd.add_output(x_add_mlp)

@pytest.mark.parametrize("num_layers", [16])
def test_pointwise_ops_benchmark(
    benchmark,
    num_layers: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    clear_cuda_cache()
    weight = torch.randn(10, 10, device="cuda", requires_grad=True)
    x = torch.randn(10, 10, device="cuda", requires_grad=True)

    with FusionDefinition() as fd:
        remat_transformer_fusion(fd, num_layers)

    if not disable_validation:
        x_add_mlp = x
        for i in range(num_layers):
            x_normed_1 = torch.nn.functional.tanh(x_add_mlp)
            attention_output = weight @ x_normed_1
            x_add_attn = attention_output + x_add_mlp
            x_normed_2 = torch.nn.functional.tanh(x_add_attn)
            mlp_output = weight @ x_normed_2
            x_add_mlp = mlp_output + x_add_attn
        fd.validate([x, weight], [x_add_mlp])

    if not disable_benchmarking:
        run_benchmark(
            benchmark,
            None,
            [x, weight],
            device="host",
            fusion_fn=partial(
                remat_transformer_fusion,
                num_layers=num_layers,
            ),
        )

