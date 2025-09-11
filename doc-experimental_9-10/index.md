## NVFuser Articles Index

A curated path from foundational topics to advanced subjects. All links are relative to this directory.

### 1) Getting Started (basics)
- Getting started: `./getting_started_tv_add_2d.md`
- Build and run samples: `./build_and_run_standalone_samples.md`
- Runtime environment issues & IR-only: `./runtime_environment_and_ir_only_mode.md`

### 2) Reading and Understanding IR
- How to read Fusion IR dumps: `./how_to_read_fusion_ir_dumps.md`
- Fusion IR anatomy & printer semantics: `./fusion_ir_anatomy_and_printer_semantics.md`
- Cache hints (cache_op): `./cache_hints_and_when_they_appear.md`

### 3) Core Concepts: TensorViews & Building IR
- Think about TensorViews (symbolic views): `./thinking_about_tensorviews.md`
- Instantiating TensorViews (TensorViewBuilder): `./instantiating_tensorviews_builder.md`
- Building an IR (FusionGuard, free functions): `./building_an_ir_anatomy_of_construction.md`
- Binding tensors & executing (FusionExecutorCache): `./binding_and_running_with_fec.md`
- Dumping IR and CUDA: `./dumping_ir_and_code.md`

### 4) Domains & Mapping
- Domains overview (IterDomain/TensorDomain; loop vs allocation): `./domains_iter_tensor_views.md`
- Domain mapping across producers/consumers & indexing: `./domain_mapping_and_indexing.md`

### 5) View Ops (reshape/squeeze/broadcast/transpose)
- View analysis & reshape flow: `./view_analysis_and_reshape_flow.md`
- Empty dimensions in reshape: `./empty_dimensions_in_reshape.md`
- What is squeeze? `./what_is_squeeze_examples.md`
- What is broadcast? `./what_is_broadcast_examples.md`
- Transpose & shape compatibility: `./transpose_and_shape_compatibility.md`

### 6) Scheduling & Loop Semantics
- Split/Merge/Reorder basics; indivisible split & predication: `./split_merge_reorder_basics.md`
- Weak vs strong correctness; neutral values: `./weak_vs_strong_correctness.md`
- Vectorization alignment & tail handling: `./vectorization_alignment_and_tail.md`
- Manual scheduling & scheduler choice heuristics: `./manual_scheduling_and_scheduler_choice.md`

### 7) Slicing, Offsets, and Indexing
- Slice via consumer offsets; compute-at & IndexCompute: `./slicing_offsets_and_indexing.md`

### 8) TMA, Tiles, and Tensor Core Transfers (advanced)
- TMA fundamentals; FTTC and efficient weak correctness: `./tma_fundamentals_fttc.md`
- Consumer schedule constraints for TMA: `./tma_consumer_schedule_constraints.md`
- LdMatrix/StMatrix inner-loop parallelization (TIDx, Vectorize): `./ldmatrix_stmatrix_inner_loop_parallelization.md`

### 9) Runtime Paths, Debugging, and Python Frontend
- Host IR JIT vs CUDA kernel JIT: `./host_ir_jit_vs_cuda_jit.md`
- Debugging with NVFUSER_DUMP: `./debugging_nvfuser_dump.md`
- Python frontend inspection APIs: `./python_frontend_inspection_apis.md`

### 10) Consolidated Guides & Q&A Map
- Implementation-oriented guide: `./implementation_oriented_guide.md`
- Q&A Topic Index (1–16 → articles): `./q_and_a_topic_index.md`


