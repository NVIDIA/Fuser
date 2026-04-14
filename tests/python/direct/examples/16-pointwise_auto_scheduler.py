"""
Example: Pointwise Auto Scheduler

This example demonstrates how to use nvFuser's automatic scheduling
for simple pointwise operations using the PointwiseScheduler.
"""

import torch
from nvfuser_direct import FusionDefinition, SchedulerType
from nvfuser_direct import schedule

inputs = [
    torch.randn(4, 4, device="cuda"),
    torch.randn(4, 4, device="cuda"),
]

with FusionDefinition() as fd:
    t0 = fd.from_pytorch(inputs[0])
    t1 = fd.from_pytorch(inputs[1])
    t2 = fd.ops.add(t0, t1)
    t3 = fd.ops.exp(t2)
    fd.add_output(t3)

    # Find compatible schedulers
    available_heuristics = schedule.find_compatible_schedulers(fd.fusion, inputs)
    print(f"Available schedulers: {available_heuristics}")
    
    # Verify that only pointwise scheduler is available
    assert len(available_heuristics) == 1
    assert set(available_heuristics) == set([SchedulerType.pointwise])
    
    # Apply pointwise scheduler
    heuristic_params = schedule.schedule(fd.fusion, SchedulerType.pointwise, inputs)

print("\n=== Fusion Math (After Auto Scheduling) ===")
print(fd.fusion.print_math())

nvf_out = fd.manual_execute(inputs, heuristic_params)
eager_out = torch.exp(inputs[0] + inputs[1])

torch.testing.assert_close(eager_out, nvf_out[0])

print("\n✓ Validation passed!")
print(f"Output shape: {nvf_out[0].shape}")
print("Automatic scheduling completed successfully!")

