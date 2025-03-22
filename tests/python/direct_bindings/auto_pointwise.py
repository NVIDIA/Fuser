# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import direct, DataType  # noqa: F401
from nvfuser import MemoryType, ParallelType, SchedulerType
from direct_fusion_definition import FusionDefinition

fd = FusionDefinition()
tv0 = direct.TensorViewBuilder().num_dims(2).contiguity(True).build()
tv1 = direct.TensorViewBuilder().num_dims(2).contiguity(True).build()

fd.add_input(tv0)
fd.add_input(tv1)
tv2 = fd.ops.add(tv0, tv1)
fd.add_output(tv2)

# copy before scheduling
fd_str = direct.translate_fusion(fd.fusion)


def nvfuser_fusion(fd: FusionDefinition) -> None:
    tv0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
    )
    tv1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
    )
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)


inputs = [
    torch.ones(4, 8, device="cuda"),
    torch.ones(4, 8, device="cuda"),
]
print(
    "Can schedule pointwise?",
    fd.schedule.can_schedule(fd.fusion, inputs, SchedulerType.pointwise),
)
print("Find schedulers", fd.schedule.find_compatible_schedulers(fd.fusion, inputs))
fd.schedule.auto_schedule(fd.fusion, inputs, SchedulerType.pointwise)
print(fd.execute(inputs, auto_schedule=False))

exec(fd_str)
func_name = "nvfuser_fusion"
with FusionDefinition() as fd_cap:
    eval(func_name)(fd_cap)

params = fd_cap.schedule.compute_heuristics(
    fd_cap.fusion, inputs, SchedulerType.pointwise
)
print(params)
fd_cap.schedule.auto_schedule(fd_cap.fusion, inputs, SchedulerType.pointwise, params)
captured_outputs = fd_cap.execute(inputs, auto_schedule=False)
print(captured_outputs)
