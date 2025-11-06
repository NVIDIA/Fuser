"""
Example: Reshape Operations

This example demonstrates how reshape operations work in nvFuser,
including how they interact with root, logical, and loop domains.
"""

import torch
from nvfuser_direct import (
    FusionDefinition,
    Merge,
    Split,
    BroadcastOp,
    SqueezeOp,
    ReshapeOp,
)

# Example 1: Basic Reshape
print("=" * 60)
print("Example 1: Basic Reshape")
print("=" * 60)

with FusionDefinition() as fd:
    tv0 = fd.define_tensor(shape=[4, 8])

    # Shape of tv0 is assumed to be [4, 8], which is then reshaped to [32]
    tv1 = fd.ops.reshape(tv0, [32])
    fd.add_output(tv1)

    # Notice that tv1 has root and logical domains. The root domain has two
    # IterDomains, whereas the logical domain consists of a single
    # IterDomain that is an output of a merge operation of the two root
    # IterDomains.
    print("\n=== Fusion Math ===")
    print(fd.fusion.print_math())

# Check if the tv1 domains are generated as expected
assert tv1.has_root()
assert len(tv1.get_logical_domain()) == 1
# In python, use type() function to check an object's class.
tv1_merge = tv1.get_logical_domain()[0].definition()
assert type(tv1_merge) is Merge
assert tv1_merge.inner() == tv1.get_root_domain()[1]
assert tv1_merge.outer() == tv1.get_root_domain()[0]

print("\n✓ Root and logical domains validated!")
print(f"  Root domain: {len(tv1.get_root_domain())} dimensions")
print(f"  Logical domain: {len(tv1.get_logical_domain())} dimension")


# Example 2: Reshape with Broadcast Domains
print("\n" + "=" * 60)
print("Example 2: Reshape with Broadcast Domains")
print("=" * 60)

with FusionDefinition() as fd:
    # Create a 3D tensor with a broadcast domain
    tv0 = fd.define_tensor(shape=[1, 2, 3])

    # tv0 is first squeezed and then reshaped and unsqueezed
    tv1 = fd.ops.reshape(tv0, [3, 2, 1])
    fd.add_output(tv1)

    print("\n=== Fusion Math ===")
    print(fd.fusion.print_math())

    # The fusion should look like:
    # tv1 = unsqueeze(reshape(squeeze(tv0)));
    assert type(tv1.definition()) is BroadcastOp
    reshape_output = tv1.definition().input(0)
    assert type(reshape_output.definition()) is ReshapeOp
    squeeze_output = reshape_output.definition().input(0)
    assert type(squeeze_output.definition()) is SqueezeOp

    assert reshape_output.has_root()
    assert len(reshape_output.get_logical_domain()) == 2
    assert type(reshape_output.get_logical_domain()[0].definition()) is Split
    reshape_output_split = reshape_output.get_logical_domain()[0].definition()
    assert reshape_output_split.outer() == reshape_output.get_logical_domain()[0]
    assert reshape_output_split.inner() == reshape_output.get_logical_domain()[1]
    assert type(reshape_output_split.input(0).definition()) is Merge
    reshape_output_merge = reshape_output_split.input(0).definition()
    assert reshape_output_merge.outer() == reshape_output.get_root_domain()[0]
    assert reshape_output_merge.inner() == reshape_output.get_root_domain()[1]

    print("\n✓ Reshape with squeeze/unsqueeze validated!")

    # So far, the fusion has transformations as part of its definition. It can
    # be further extended with scheduling transformations.
    reshape_output.merge(0, 1)
    reshape_output.split(0, 128)

    assert type(reshape_output.get_loop_domain()[0].definition()) is Split
    assert (
        reshape_output.get_loop_domain()[0].definition().inner()
        == reshape_output.get_loop_domain()[1]
    )

    # Here's how we propagate the transformations of reshape_output to all
    # other tensors in the fusion
    fd.sched.transform_like(reshape_output)

    # Now, all tensors, including those before the reshape op, should be
    # transformed to 2D tensors with an inner domain of extent 128.
    print("\n=== Fusion Math (After scheduling) ===")
    print(fd.fusion.print_math())

    # Notice that all transformations of the reshape tensor, including both the
    # reshape and scheduling transformations, are propagated.
    
    # Note that all the transformations of squeeze_output are scheduling
    # transformations, thus it should not have a root domain
    assert not squeeze_output.has_root()
    
    print("\n✓ Transform propagation validated!")

print("\n" + "=" * 60)
print("All reshape examples completed successfully!")
print("=" * 60)

