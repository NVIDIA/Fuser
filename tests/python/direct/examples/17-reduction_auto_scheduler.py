"""
Example: Reduction Auto Scheduler

This example demonstrates how to use nvFuser's automatic scheduling
for reduction operations using the ReductionScheduler.
"""

import torch
from nvfuser_direct import FusionDefinition, SchedulerType
from nvfuser_direct import schedule

inputs = [
    torch.randn(4, 4, device="cuda"),
]

with FusionDefinition() as fd:
    t0 = fd.from_pytorch(inputs[0])
    t1 = fd.ops.sum(t0, dims=[1])
    t2 = fd.ops.exp(t1)
    fd.add_output(t2)

    # Test that pointwise scheduler is not available for reduction
    pointwise_status, error_msg = schedule.can_schedule(
        fd.fusion, SchedulerType.pointwise, inputs
    )
    assert not pointwise_status
    print("Pointwise scheduler status:", pointwise_status)
    print(f"Error message: {error_msg.strip()}\n")

    # Find compatible schedulers
    available_heuristics = schedule.find_compatible_schedulers(fd.fusion, inputs)
    print(f"Available schedulers: {available_heuristics}")
    
    # Verify that only reduction scheduler is available
    assert len(available_heuristics) == 1
    assert set(available_heuristics) == set([SchedulerType.reduction])
    
    # Apply reduction scheduler
    heuristic_params = schedule.schedule(fd.fusion, SchedulerType.reduction, inputs)

print("\n=== Fusion Math (After Auto Scheduling) ===")
print(fd.fusion.print_math())

nvf_out = fd.manual_execute(inputs, heuristic_params)
eager_out = torch.exp(inputs[0].sum(1))

torch.testing.assert_close(eager_out, nvf_out[0])

print("\n✓ Validation passed!")
print(f"Output shape: {nvf_out[0].shape}")
print("Automatic scheduling completed successfully!")

