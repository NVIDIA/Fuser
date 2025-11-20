# TODO: Enable IdModel Indexing for C++ Tests

This file tracks the progress of enabling IdModel indexing across all C++ test files.

**Goal**: Add `EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"})` to all test files to enable the new IdModel-based indexing system.

## Progress Summary

- **Total test files**: 124
- **Completed**: 28 (22.6%)
- **Remaining**: 96 (77.4%)

## Implementation Pattern

For test fixtures:
```cpp
class MyTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};
```

For individual tests:
```cpp
TEST_F(NVFuserTest, MyTestName) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  // test code...
}
```

---

## Completed Files ✓ (28)

These files already have IdModel enabled:

- [x] test_argsort.cpp
- [x] test_bfs.cpp
- [x] test_circular_buffering.cpp
- [x] test_circular_buffering_ping_pong.cpp
- [x] test_combined_inner_outer_reduction.cpp
- [x] test_compute_with.cpp
- [x] test_contiguity_id_model.cpp
- [x] test_gather.cpp
- [x] test_gpu3.cpp
- [x] test_greedy.cpp
- [x] test_indexing.cpp
- [x] test_indexing_advanced.cpp
- [x] test_index_put.cpp
- [x] test_index_select.cpp
- [x] test_layout_op.cpp
- [x] test_loop_domain_scheduling.cpp
- [x] test_matmul.cpp
- [x] test_memory.cpp
- [x] test_moe.cpp
- [x] test_pointwise.cpp
- [x] test_reshape.cpp
- [x] test_resize.cpp
- [x] test_rng.cpp
- [x] test_scatter.cpp
- [x] test_select.cpp
- [x] test_topk.cpp
- [x] test_transpose.cpp
- [x] utils.h (BlackwellBase, TmaBase, HopperBase fixtures)

---

## Remaining Files (96)

### Core IR and Analysis Tests (18)

- [ ] test_abstract_tensor.cpp
- [ ] test_alias.cpp
- [ ] test_alias_analysis.cpp
- [ ] test_compute_at_map.cpp
- [ ] test_evaluator.cpp
- [ ] test_expr_simplifier.cpp
- [ ] test_expr_sort.cpp
- [ ] test_fusion_hash.cpp
- [ ] test_id_model.cpp
- [ ] test_interval_analysis.cpp
- [ ] test_iter_visitor.cpp
- [ ] test_linked_hash_map.cpp
- [ ] test_meta.cpp
- [ ] test_mutator.cpp
- [ ] test_polymorphic_value.cpp
- [ ] test_statement_guard.cpp
- [ ] test_utils.cpp
- [ ] test_ca_root_domain_map.cpp

### GPU Tests (3)

- [ ] test_gpu1.cpp
- [ ] test_gpu2.cpp
- [ ] test_external_src.cpp

### Scheduling and Transformation Tests (9)

- [ ] test_allocation_domain.cpp
- [ ] test_allocation_order_inference.cpp
- [ ] test_inlining.cpp
- [ ] test_loop_rotation.cpp
- [ ] test_replay.cpp
- [ ] test_move_pad.cpp
- [ ] test_move_repeat_forward.cpp
- [ ] test_move_split_cat.cpp
- [ ] test_remove_bcast_squeeze.cpp

### Reduction and Normalization Tests (6)

- [ ] test_outer_reduction.cpp
- [ ] test_persistent_buffer.cpp
- [ ] test_reduction.cpp
- [ ] test_reduction_pointwise.cpp
- [ ] test_serial_gridreduce.cpp
- [ ] test_welford.cpp

### Operations Tests (13)

- [ ] test_driver_api.cpp
- [ ] test_dynamic_transform.cpp
- [ ] test_embedding_node.cpp
- [ ] test_math_opt.cpp
- [ ] test_no_op.cpp
- [ ] test_predicate_elimination.cpp
- [ ] test_preseg_passes.cpp
- [ ] test_remove_trivial_ops.cpp
- [ ] test_rope.cpp
- [ ] test_scalar_hoisting.cpp
- [ ] test_sdpa_node.cpp
- [ ] test_unary.cpp
- [ ] test_vectorization.cpp

### Matmul and MMA Tests (5)

- [ ] test_cutlass_scheduler.cpp
- [ ] test_matmul_aten_evaluation.cpp
- [ ] test_matmul_sass.cpp
- [ ] test_matmul_scheduler.cpp
- [ ] test_mma.cpp
- [ ] test_translate_mma.cpp

### Memory and Synchronization Tests (6)

- [ ] test_mbarrier.cpp
- [ ] test_overlap.cpp
- [ ] test_smem_reuse.cpp
- [ ] test_stream.cpp
- [ ] test_swizzle.cpp
- [ ] test_tensor_factories.cpp

### Specialized Algorithm Tests (5)

- [ ] test_argsort_device_func.cpp
- [ ] test_scan.cpp
- [ ] test_scan_device_func.cpp
- [ ] test_topk_device_func.cpp
- [ ] test_cluster.cpp
- [ ] cluster_runtime_test/test_cluster_device_func.cpp

### Multidevice Tests (12)

- [ ] test_multidevice_communications.cpp
- [ ] test_multidevice_communicator.cpp
- [ ] test_multidevice_host_ir.cpp
- [ ] test_multidevice_host_ir_overlap.cpp
- [ ] test_multidevice_ipc.cpp
- [ ] test_multidevice_lower_communication.cpp
- [ ] test_multidevice_matmul.cpp
- [ ] test_multidevice_pipeline.cpp
- [ ] test_multidevice_sharding.cpp
- [ ] test_multidevice_stream_parallel_type.cpp
- [ ] test_multidevice_transformer.cpp
- [ ] test_multidevice_tutorial.cpp

### Host IR Tests (5)

- [ ] test_host_ir_evaluator.cpp
- [ ] test_host_ir_integration.cpp
- [ ] test_host_ir_jit.cpp
- [ ] test_host_irs.cpp
- [ ] test_host_ir_stream_lowering.cpp

### Kernel Database Tests (3)

- [ ] kernel_db/test_nvfuser_kernel_db_open.cpp
- [ ] kernel_db/test_nvfuser_kernel_db_query.cpp
- [ ] kernel_db/test_nvfuser_kernel_db_write.cpp

### Miscellaneous Tests (11)

- [ ] test_exceptions.cpp
- [ ] test_fusion_profiler.cpp
- [ ] test_iostream.cpp
- [ ] test_low_precision_recipe.cpp
- [ ] test_resharding.cpp
- [ ] test_runtime.cpp
- [ ] test_segmentation.cpp
- [ ] test_sharding.cpp
- [ ] test_tmem.cpp
- [ ] test_tutorial.cpp

---

## Testing Approach

When enabling IdModel for each file:

1. **Add the EnableOptionsGuard** to the test fixture or individual tests
2. **Run the specific test** to ensure it passes:
   ```bash
   bin/test_nvfuser --gtest_filter=TestName*
   ```
3. **Fix any failures** that occur due to IdModel indexing differences
4. **Mark the file as complete** in this TODO list

## Notes

- Some tests may require adjustments beyond just adding the EnableOptionsGuard
- Tests that heavily rely on legacy indexing behavior may need more substantial updates
- Device function tests (*_device_func.cpp) may have special considerations
- Multidevice tests require MPI environment to run properly

## Verification Command

To check current status:
```bash
cd /tmp/nvfuser/tests/cpp
for file in test_*.cpp; do
  if grep -q "EnableOptionsGuard.*IdModel\|EnableOption::IdModel" "$file" 2>/dev/null; then
    echo "✓ $file"
  else
    echo "✗ $file"
  fi
done
```

---

Last updated: 2025-11-19
