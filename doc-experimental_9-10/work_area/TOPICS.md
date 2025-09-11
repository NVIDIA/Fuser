# NVFuser Workshop Topics Index (Stage 1: Exhaustive Topic List)

This file catalogs topics to develop into rich articles (Stage 2). Links are relative to this folder.

## Getting Started and Environment

- Getting started: first standalone nvFuser program (tv_add_2d)
  - Sources: [../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_1.cpp](../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_1.cpp), [../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp](../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp)
  - Logs: [../doc-bot/experimenting/8-22-2025/chat_log.md](../doc-bot/experimenting/8-22-2025/chat_log.md)
- Building and running standalone samples
  - Scripts: [../doc-bot/how_to_build](../doc-bot/how_to_build), [../doc-bot/how_to_run](../doc-bot/how_to_run), [../doc-bot/setup_libs_to_run](../doc-bot/setup_libs_to_run)
  - Workarounds: [../doc-bot/experimenting/9-3-2025/workarounds](../doc-bot/experimenting/9-3-2025/workarounds)
  - Logs: [../doc-bot/experimenting/9-3-2025/log](../doc-bot/experimenting/9-3-2025/log)
- Runtime environment issues and IR-only mode
  - Notes: [../doc-bot/experimenting/9-3-2025/workarounds](../doc-bot/experimenting/9-3-2025/workarounds)

## Fusion IR and Tensor Fundamentals

- Fusion IR anatomy and printer semantics (symbols, memory spaces, Set vs Set.Permute)
  - Guide: [../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md](../doc-bot/experimenting/8-26-2025/Fusion_IR_Anatomy.md)
  - Sample: [../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp](../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp)
  - Insights: [../doc-bot/experimenting/8-26-2025/key_insights.md](../doc-bot/experimenting/8-26-2025/key_insights.md)
- How to read Fusion IR dumps (e.g., `T#_{mem}_{dtype}` labeling)
  - Notes: [../doc-bot/experimenting/8-26-2025/key_insights.md](../doc-bot/experimenting/8-26-2025/key_insights.md)
- TensorView, IterDomain, TensorDomain; logical vs allocation vs loop domains
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topics 1–2, 12–14)
  - Code: [../csrc/ir/nodes.cpp](../csrc/ir/nodes.cpp)
- Domain mapping across producers/consumers and its role in indexing
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topic 2)
  - Code (if present): [../csrc/index_compute.cpp](../csrc/index_compute.cpp), [../csrc/transform_iter.cpp](../csrc/transform_iter.cpp)
- Cache hints (`cache_op`) and when they appear
  - Notes: [../doc-bot/experimenting/8-26-2025/key_insights.md](../doc-bot/experimenting/8-26-2025/key_insights.md), [../doc-bot/experimenting/8-26-2025/what_AI_learned](../doc-bot/experimenting/8-26-2025/what_AI_learned)

## View Ops: Reshape / Squeeze / Broadcast / Transpose

- What is a ViewAnalysis and how is it used
  - Source: [../doc-bot/experimenting/9-3-2025/chat_log_3.md](../doc-bot/experimenting/9-3-2025/chat_log_3.md), [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)
  - Code: [../csrc/transform_view.h](../csrc/transform_view.h), [../csrc/transform_view.cpp](../csrc/transform_view.cpp)
- Reshape flow: analyzeView → applyViewTransforms → ViewOp
  - Notes: [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)
  - Code: [../csrc/ops/alias.cpp](../csrc/ops/alias.cpp), [../csrc/transform_view.cpp](../csrc/transform_view.cpp), [../csrc/ir/nodes.cpp](../csrc/ir/nodes.cpp)
- "Empty" dimensions in reshape: semantics and implications
  - Notes: [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)
  - Code: [../csrc/ops/alias.cpp](../csrc/ops/alias.cpp)
- What is a "squeeze"? With examples
  - Notes: [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)
  - Code: [../csrc/ops/alias.cpp](../csrc/ops/alias.cpp)
- What is a "broadcast"? With examples
  - Notes: [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)
  - Code: [../csrc/ops/alias.cpp](../csrc/ops/alias.cpp)
- Transpose and shape compatibility for elementwise ops
  - Sample: [../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp](../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp)
  - Insights: [../doc-bot/experimenting/8-26-2025/key_insights.md](../doc-bot/experimenting/8-26-2025/key_insights.md)

## Building and Executing IR

- How to think about TensorViews (symbolic view, not data)
  - Notes: [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)
- Instantiating TensorViews (TensorViewBuilder); cannot import from at::Tensor
  - Notes: same file; Code: [../csrc/ops/alias.cpp](../csrc/ops/alias.cpp)
- Binding physical tensors and executing via FusionExecutorCache
  - Notes: same file; Runtime: [../python/nvfuser](../python/nvfuser) (if present)
- Anatomy of building an IR (FusionGuard, free functions)
  - Sample: [../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp](../doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp)
- How we dump IR and code: printFusion, printTransforms, getMostRecentScheduledIr, getMostRecentCode
  - Notes: [../doc-bot/experimenting/9-3-2025/key_insights.md](../doc-bot/experimenting/9-3-2025/key_insights.md)

## Scheduling and Loop/Memory Semantics

- Split/merge/reorder basics; divisible vs indivisible split; predication
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topic 3)
  - Guide: [../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md](../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md)
- Vectorization alignment and tail handling (prologue/main/epilogue)
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topic 16)
- Weak vs strong correctness; hole semantics; neutral values
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topic 5)
- Manual scheduling and scheduler choice heuristics
  - Samples: [../doc-bot/experimenting/9-3-2025/manual_schedule_SAMPLE_2.cpp](../doc-bot/experimenting/9-3-2025/manual_schedule_SAMPLE_2.cpp), [../doc-bot/experimenting/9-3-2025/scheduler_choice_SAMPLE_4.cpp](../doc-bot/experimenting/9-3-2025/scheduler_choice_SAMPLE_4.cpp), [../doc-bot/experimenting/9-3-2025/custom_op_schedule_SAMPLE_3.cpp](../doc-bot/experimenting/9-3-2025/custom_op_schedule_SAMPLE_3.cpp)
  - Logs: [../doc-bot/experimenting/9-3-2025/chat_log_1.md](../doc-bot/experimenting/9-3-2025/chat_log_1.md), [../doc-bot/experimenting/9-3-2025/chat_log_2.md](../doc-bot/experimenting/9-3-2025/chat_log_2.md), [../doc-bot/experimenting/9-3-2025/chat_log_3.md](../doc-bot/experimenting/9-3-2025/chat_log_3.md)

## TMA, Tiles, and Tensor Core Transfers

- TMA fundamentals; FTTC (impossibility of strong correctness) and weak-correctness strategies
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topics 4, 5)
- Consumer schedule constraints for TMA (separation of tile vs non-tile transforms; whole-tile allocation; swizzle constraints)
  - Q&A: same file (Topic 6)
- LdMatrix/StMatrix integration and inner-loop parallelization (TIDx, Vectorize)
  - Q&A: same file (Topic 7)

## Runtime Paths, Debugging, and Python Frontend

- Host IR JIT vs CUDA kernel JIT
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topic 8)
  - Guide: [../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md](../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md)
- Debugging with NVFUSER_DUMP and related toggles
  - Q&A: same file (Topic 9)
- Python frontend inspection APIs (FusionDefinition)
  - Q&A: same file (Topic 11)

## Slicing, Offsets, and Indexing

- Slice via consumer offsets; compute-at and IndexCompute
  - Q&A: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md) (Topics 15–16)

## Consolidated Guides and References

- Implementation-oriented guide (overview of transforms, TMA, debugging, runtime)
  - Guide: [../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md](../doc-bot/experimenting/8-28-2025/nvfuser_implementation_guide.md)
- Summary artifacts and planning (for cross-links during drafting)
  - Stage docs: [../doc-bot/experimenting/8-28-2025/stage1_information_inventory.md](../doc-bot/experimenting/8-28-2025/stage1_information_inventory.md), [../doc-bot/experimenting/8-28-2025/stage3_core_concepts.md](../doc-bot/experimenting/8-28-2025/stage3_core_concepts.md), [../doc-bot/experimenting/8-28-2025/stage4_content_mapping.md](../doc-bot/experimenting/8-28-2025/stage4_content_mapping.md), [../doc-bot/experimenting/8-28-2025/stage5_outline_scaffolding.md](../doc-bot/experimenting/8-28-2025/stage5_outline_scaffolding.md)

## Future Work and Open Questions (to track, not to summarize away)

- Future work (8-26)
  - [../doc-bot/experimenting/8-26-2025/future_work.md](../doc-bot/experimenting/8-26-2025/future_work.md)
- Future work (9-3)
  - [../doc-bot/experimenting/9-3-2025/future_work.md](../doc-bot/experimenting/9-3-2025/future_work.md)

## Q&A Topic List (source of many deep-dive articles)

Each bullet corresponds to a standalone article candidate; all sourced from: [../doc-bot/experimenting/8-28-2025/q_and_a.md](../doc-bot/experimenting/8-28-2025/q_and_a.md)

1) Big-picture nvFuser flow (frontend → IR → schedule → CUDA/Host IR)
2) Domains overview: IterDomain/TensorDomain; loop vs allocation; DomainMap
3) Split semantics: divisible vs indivisible; predicates and performance
4) TMA FTTC: when strong correctness is impossible; efficient weak correctness
5) Weak vs strong correctness; neutral values and reductions
6) TMA consumer constraints: separate tile vs non-tile; contiguity; whole-tile allocation
7) LdMatrix/StMatrix inner-loop parallelization (TIDx, Vectorize)
8) Host IR JIT vs CUDA JIT: when and why
9) Debugging: NVFUSER_DUMP modes (cuda_to_file, segmented_fusion, etc.)
10) Benefit of separating tile vs non-tile transforms (TMA indexing/validity)
11) Python frontend inspection APIs: fusion_ir(), last_cuda_code()
12) TensorDomain views: root/logical, allocation, loop
13) Who determines loop domain; APIs to retrieve domains
14) Recap of the three domain views and purposes
15) Slice consumer offsets without materializing sub-tensors
16) Vectorization tail handling for misaligned slices

---

If any topic is missing or needs re-grouping, we will adjust before drafting Stage 2 articles.
