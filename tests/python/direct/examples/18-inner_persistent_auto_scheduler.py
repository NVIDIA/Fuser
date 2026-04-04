"""
Example: Inner Persistent Auto Scheduler

This example demonstrates how to use nvFuser's automatic scheduling
for normalization kernels using the InnerPersistentScheduler.
"""

import torch
from nvfuser_direct import FusionDefinition, SchedulerType, DataType
from nvfuser_direct import schedule

tensor_size = 4
inputs = [torch.randn(tensor_size, tensor_size, device="cuda")]

with FusionDefinition() as fd:
    t0 = fd.from_pytorch(inputs[0])
    s0 = fd.define_scalar(1e-6, dtype=DataType.Double)
    norm_const = fd.define_scalar(tensor_size, dtype=DataType.Int)

    bcast_sum0 = fd.ops.sum(t0, dims=[-1], keepdim=True)
    mean = fd.ops.div(bcast_sum0, norm_const)

    diff = fd.ops.sub(t0, mean)
    diff_sq = fd.ops.mul(diff, diff)
    bcast_sum1 = fd.ops.sum(diff_sq, dims=[-1], keepdim=True)
    var = fd.ops.div(bcast_sum1, norm_const)

    t0_diff = fd.ops.sub(t0, mean)
    var_eps = fd.ops.sqrt(fd.ops.add(var, s0))
    t0_norm = fd.ops.div(t0_diff, var_eps)
    fd.add_output(t0_norm)

    # Find compatible schedulers
    available_heuristics = schedule.find_compatible_schedulers(fd.fusion, inputs)
    print(f"Available schedulers: {available_heuristics}")
    
    # Verify that only inner_persistent scheduler is available
    assert len(available_heuristics) == 1
    assert set(available_heuristics) == set([SchedulerType.inner_persistent])
    
    # Apply inner_persistent scheduler
    heuristic_params = schedule.schedule(fd.fusion, SchedulerType.inner_persistent, inputs)

print("\n=== Fusion Math (After Auto Scheduling) ===")
print(fd.fusion.print_math())

nvf_out = fd.manual_execute(inputs, heuristic_params)
var, mean = torch.var_mean(inputs[0], dim=-1, correction=0, keepdim=True)
eager_out = (inputs[0] - mean) / torch.sqrt(var + 1e-6)

torch.testing.assert_close(eager_out, nvf_out[0])

print("\n✓ Validation passed!")
print(f"Output shape: {nvf_out[0].shape}")
print("Normalization with automatic scheduling completed successfully!")

