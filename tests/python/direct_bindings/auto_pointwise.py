# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import direct
from nvfuser import MemoryType, ParallelType, SchedulerType
from direct_fusion_definition import FusionDefinition

fd = FusionDefinition()
tv0 = direct.TensorViewBuilder().num_dims(2).contiguity(True).build()
tv1 = direct.TensorViewBuilder().num_dims(2).contiguity(True).build()

fd.add_input(tv0)
fd.add_input(tv1)
tv2 = fd.ops.add(tv0, tv1)
fd.add_output(tv2)

inputs = [
    torch.ones(4, 8, device="cuda"),
    torch.ones(4, 8, device="cuda"),
]
print(fd.schedule.can_schedule(fd.fusion, inputs, SchedulerType.pointwise))
print(fd.schedule.find_compatible_schedulers(fd.fusion, inputs))
fd.schedule.schedule(fd.fusion, inputs, SchedulerType.pointwise)
print(fd.execute(inputs, auto_schedule=False))
