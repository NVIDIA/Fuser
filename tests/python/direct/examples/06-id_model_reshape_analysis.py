"""
Example: IdModel for Reshape Analysis

This example demonstrates how to use IdModel for analyzing equivalence
of reshape operations in nvFuser.
"""

import torch
from nvfuser_direct import FusionDefinition, IdMappingMode
from nvfuser_direct import idm

print("=" * 60)
print("IdModel Reshape Analysis Example")
print("=" * 60)
print("""
IdModel is a powerful tool for analyzing equivalence relationships
between IterDomains in a fusion. This example shows how to use it
to determine if reshape operations on different tensors are equivalent.
""")

with FusionDefinition() as fd:
    # Use the static reshape to avoid reshape concretization.
    tv0 = fd.define_tensor(shape=[10, 20])
    tv1 = fd.define_tensor(shape=[10, 20])

    # While the reshape operations are equivalent, we do not know if the two
    # inputs are the same. There is not an operation allowing us to infer
    # equivalence. e.g., tv0 + tv1.
    tv2 = fd.ops.reshape(tv0, [20, 10])
    tv3 = fd.ops.reshape(tv1, [20, 10])
    fd.add_output(tv2)
    fd.add_output(tv3)

print("\n=== Fusion Math ===")
print(fd.fusion.print_math())

# Build the IdModel
id_model = idm.IdModel(fd.fusion)
exact_graph = id_model.maybe_build_graph(IdMappingMode.exact)

print("\n=== IdModel ===")
print(id_model)
print("\n=== Exact Graph ===")
print(exact_graph)
print("\n=== Disjoint Val Sets ===")
print(exact_graph.disjoint_val_sets())

# As mentioned above, we do not know any relationship between tv0 and tv1.
# They should not be mapped in exact graph.
print("\n--- Checking tv0 and tv1 mappings ---")
assert len(tv0.get_logical_domain()) == len(tv1.get_logical_domain())
for i, (tv0_id, tv1_id) in enumerate(zip(tv0.get_logical_domain(), tv1.get_logical_domain())):
    are_mapped = exact_graph.disjoint_val_sets().strict_are_mapped(tv0_id, tv1_id)
    print(f"tv0[{i}] and tv1[{i}] mapped: {are_mapped}")
    assert not are_mapped

# Thus, the outputs of the reshape ops are not mapped either
print("\n--- Checking tv2 and tv3 mappings (before manual mapping) ---")
assert len(tv2.get_loop_domain()) == len(tv3.get_loop_domain())
for i, (tv2_id, tv3_id) in enumerate(zip(tv2.get_loop_domain(), tv3.get_loop_domain())):
    are_mapped = exact_graph.disjoint_val_sets().strict_are_mapped(tv2_id, tv3_id)
    print(f"tv2[{i}] and tv3[{i}] mapped: {are_mapped}")
    assert not are_mapped

# Now, suppose we can say the inputs are exactly mapped. We can manually
# add mappings:
print("\n--- Manually mapping tv0 and tv1 ---")
for tv0_id, tv1_id in zip(tv0.get_logical_domain(), tv1.get_logical_domain()):
    exact_graph.map_vals(tv0_id, tv1_id)
print("✓ Inputs manually mapped")

# Now, tv2 and tv3 should be fully mapped, including their root,
# intermediate and loop domains.

# Check the root domains.
print("\n--- Checking tv2 and tv3 root domain mappings ---")
assert len(tv2.get_root_domain()) == len(tv3.get_root_domain())
for i, (tv2_id, tv3_id) in enumerate(zip(tv2.get_root_domain(), tv3.get_root_domain())):
    are_mapped = exact_graph.disjoint_val_sets().strict_are_mapped(tv2_id, tv3_id)
    print(f"tv2_root[{i}] and tv3_root[{i}] mapped: {are_mapped}")
    assert are_mapped

# The reshape consists of a merge and split. The output of the merge should
# be mapped as well
print("\n--- Checking intermediate merge outputs ---")
tv2_merge_out = tv2.get_root_domain()[0].uses()[0].output(0)
tv3_merge_out = tv3.get_root_domain()[0].uses()[0].output(0)
are_mapped = exact_graph.disjoint_val_sets().strict_are_mapped(
    tv2_merge_out, tv3_merge_out
)
print(f"Merge outputs mapped: {are_mapped}")
assert are_mapped

# The next operation is split. Its outputs, which are the loop domains,
# should be mapped too.
print("\n--- Checking tv2 and tv3 loop domain mappings (after manual mapping) ---")
for i, (tv2_id, tv3_id) in enumerate(zip(tv2.get_loop_domain(), tv3.get_loop_domain())):
    are_mapped = exact_graph.disjoint_val_sets().strict_are_mapped(tv2_id, tv3_id)
    print(f"tv2[{i}] and tv3[{i}] mapped: {are_mapped}")
    assert are_mapped

print("\n" + "=" * 60)
print("✓ All IdModel analysis checks passed!")
print("=" * 60)

