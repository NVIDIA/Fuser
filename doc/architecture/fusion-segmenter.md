# NVFuser Fusion Segmenter Development Notes

## Goals and Motivation

-   **Why Segment?** Fusing an entire complex computation into a single GPU kernel is often impractical or suboptimal due to:
    -   Operations unsupported by the GPU kernel code generator (e.g., certain communications, host-side logic, specific ATen calls handled by `ExprEval`).
    -   Hard segmentation hints (`LoadStoreOpType::SegmenterSet`) explicitly requesting a break.
    -   Complex control flow or graph structures that are difficult to schedule efficiently as one unit.
    -   Different parts of the graph having conflicting optimal scheduling strategies (e.g., reductions vs. pointwise ops).
    -   Excessive register pressure or shared memory usage in a monolithic kernel.
-   **Segmentation Addresses This By:** Breaking the computation graph into smaller, manageable sub-graphs (`SegmentedGroup`s). Each segment can then be independently scheduled and potentially compiled into a separate, optimized GPU kernel (`SchedulerType` like Pointwise, Reduction) or executed on the host via expression evaluation (`SchedulerType::ExprEval`).

## Core Principle: Schedulability-Driven Partitioning & Input/Output Boundaries

-   **Schedulability:** Segmentation is not arbitrary graph cutting. The primary constraint is that **each final `SegmentedGroup` must be schedulable** by a known mechanism (e.g., Pointwise heuristic, Reduction heuristic, Persistent heuristic, or the `ExprEval` executor). The merging process (`SegmentCandidateFinder`) validates each potential merge using `tryMerge` which checks schedulability via `Schedule::proposeHeuristics`.
-   **Input/Output Boundary Principle:** A fundamental aspect of how nvFuser processes `Fusion` objects (including those generated for segments) is the significance of marked inputs and outputs:
    -   **Inputs:** When a `Val` is marked as an input to a `Fusion` (via `Fusion::addInput`), subsequent analysis (like dependency checking, scheduling, code generation) treats it as externally provided. The traversal of dependencies **stops** at these inputs; their defining expressions *within that `Fusion` object* (if any exist due to copying) are effectively ignored for the purpose of executing that `Fusion`.
    -   **Outputs:** Similarly, marking `Val`s as outputs (via `Fusion::addOutput`) defines the termination points for required computation within that `Fusion`. Analysis typically proceeds backward from outputs to determine necessary expressions. Expressions not contributing to the marked outputs are generally ignored.
-   `Schedule::proposeHeuristics` determines the appropriate `SchedulerType` (GPU heuristic or `ExprEval`).
-   The fundamental output is a partitioning of the original `Fusion` into a DAG of **schedulable** execution units (GPU kernels or host-evaluated segments), where the boundaries are correctly defined by the inputs/outputs of each segment's generated `Fusion`.

## Key Performance Tradeoffs

Segmentation involves balancing several competing performance factors:

-   **Kernel Launch Overhead:** More GPU kernel segments generally mean more distinct kernels to launch, adding CPU overhead. Host (`ExprEval`) segments have different overhead characteristics.
-   **Memory Bandwidth (Inter-Segment):** Segment boundaries (between GPU segments or between GPU and host) usually imply intermediate tensor results being written to and read from global memory. Fewer boundaries often reduce this I/O. Host segments reading/writing tensors invoke ATen operations with their own memory characteristics.
-   **Scheduling Efficiency:** Smaller, more specialized GPU segments might allow for more optimal scheduling heuristics tailored to that segment's operations. Host segments offload scalar or specific ATen compute from the GPU.
-   **Recomputation vs. Memory I/O:** Sometimes, recomputing cheap operations (like simple unary ops via Input Forwarding) within multiple subsequent segments is faster than paying the memory cost to materialize and transfer the intermediate result once.
-   **Potential Tradeoff (Dimensionality - Currently Less Explicitly Optimized):** The size of data transferred at segment boundaries can be affected by dimensionality changes. Ideally, boundaries would occur *after* operations that reduce data size (like reductions) and *before* operations that increase data size (like broadcasts) to minimize memory traffic. While not the primary driver now, this influences the benefit of strategies like `CombineReductions`.

## Conceptual Configuration

-   The segmentation process includes several optional passes or strategies controlled by `SegmentCandidateFinderOptions`.
-   These passes (e.g., `run_combine_reductions`, `run_translate_welford`, `run_merge_up_and_down_cast`) apply specific heuristics to merge groups based on known beneficial patterns *before* or *during* the main merging loop, influencing the final segmentation outcome.

## High-Level Workflow Summary

The `SegmentCandidateFinder` class orchestrates the segmentation process:

1.  **Input:** Takes an original `Fusion` DAG.
2.  **Initial Segmentation (`buildInitialSegments`):** Traverses the `Fusion`'s expressions. Creates fine-grained `SegmentedGroup`s for **each expression** (including scalar and tensor operations) not handled by input forwarding. Establishes initial `SegmentedEdge`s based on `Val` dependencies (including scalar dependencies).
3.  **Input/Scalar Handling Prep:** Applies strategies like Input Forwarding (for cheap unary ops near inputs) and Up-Cast Privatization to prepare the graph for merging.
4.  **Optional Pre-Merge Passes:** Executes specialized merging passes based on `SegmentCandidateFinderOptions` if enabled (e.g., `TranslateApplicableWelford`, `MergeUpAndDownCast`, `CombineReductions`). These passes group operations based on specific known patterns.
5.  **Iterative Merging (`run_herrmann_merge`, `run_final_merge`):** The core merging loop. Iteratively identifies candidate pairs or sets of adjacent `SegmentedGroup`s to merge. **Crucially, each potential merge is validated using `tryMerge` (which calls `Schedule::proposeHeuristics`) to ensure the resulting combined group is schedulable by a known scheduler (GPU heuristic or `ExprEval`).** Merging proceeds only if schedulable.
6.  **Finalization (`finalize`, `resolveForwardedInputs`):** Cleans up the graph. Resolves previously forwarded inputs. Finalizes segment boundary metadata (`input_vals_`, `output_vals_`) within each `SegmentedGroup`. **Then, for each segment, `SegmentedFusion::makeFusion` creates a standalone `Fusion` object. It does this by copying the complete fusion's IR and then explicitly marking the correct boundary `Val`s (determined from the `SegmentedGroup`'s metadata) as inputs and outputs of the new `Fusion`. This leverages the Input/Output Boundary Principle: downstream processing ignores expressions outside the marked input-to-output paths, effectively isolating the segment's logic without needing explicit expression pruning.**
    > **Implementation Note on `makeFusion` Cloning:** The `IrCloner` object used within `makeFusion` maintains a map from original statements to cloned statements. The initial `Fusion::copy` populates this map for all statements. Subsequent calls to `cloner.clone(original_statement)` within the same `makeFusion` call act as lookups in this map, returning the already-created clone rather than performing redundant cloning.
7.  **Output:** Produces a `SegmentedFusion` object containing the final DAG of schedulable `SegmentedGroup`s, ready for execution scheduling and code generation.

## High-Level Summaries (TLDRs)

-   **TLDR: Edge System:** Manages data dependencies (scalar and tensor) between segments using `SegmentedEdge` objects and centralized helper functions. Ensures correct graph structure and data flow during merging and finalization.
-   **TLDR: Handling Specific Ops:** Cheap unary ops near inputs are often recomputed ("forwarded"). Up-casts are handled to encourage lower-precision boundaries. Segmentation hints (`SegmenterSet`) force boundaries often leading to `ExprEval` segments. Scalar ops are grouped initially and merged like tensor ops; their final execution target (`GPU` vs. `ExprEval`) is determined by the scheduler assigned to their final segment.
-   **TLDR: Precision at Boundaries:** When enabled (`IoToLowerPrecision`), the segmenter attempts to insert casts such that intermediate tensors involved in mixed-precision paths are transferred between segments in lower precision (Half/BFloat16), reducing memory bandwidth. This process is targeted based on final `Fusion` outputs and subject to operation constraints.
-   **TLDR: `ExprEval` Segmentation:** Segments containing only scalar operations or specific ops like `SegmenterSet` are typically assigned `SchedulerType::ExprEval`. This means they execute on the host (potentially calling ATen ops for tensors). From the perspective of *GPU* segments, an `ExprEval` segment acts like any other dependency boundary – values produced by `ExprEval` become inputs to the GPU kernel, preventing computation duplication.

-----

## Detailed Explanations

### Edge System: Tracking Dependencies

-   **Purpose:** The edge system is crucial for representing and maintaining data dependencies (both scalar and tensor) between `SegmentedGroup`s throughout the segmentation process. It ensures that as groups are merged, the correct data flow is preserved.
-   **Representation:**
    -   Dependencies are modeled using `SegmentedEdge` objects. Each edge connects a producer group (`from`) to a consumer group (`to`) and specifies the exact intermediate `Val` (scalar or tensor) being passed.
    -   Multiple edges can exist between the same two groups if multiple distinct `Val`s are passed.
    -   Each `SegmentedGroup` maintains lists of its incoming (`producer_edges`) and outgoing (`consumer_edges`) connections.
-   **Role in Merging:**
    -   Edges are used to identify neighboring groups and potential merge candidates.
    -   When groups are merged, the edge system logic determines which external edges need to be reconnected to the new, merged group. Edges *internal* to the merged set are discarded.
    -   Centralized helper functions (like `connectGroups`, `removeEdge`, `getEdgesBetween`) are used internally by the segmenter to ensure these modifications are done consistently and correctly, preserving the DAG structure.
-   **Data Structures:** Edges are managed centrally within the `SegmentedFusion` object, while groups hold references to their relevant producer/consumer edges. Input/output `Val`s within groups use `VectorOfUniqueEntries` for efficient handling.

### Handling of Specific Operation Types

#### Unary Operations

-   **Input Forwarding:** Chains of simple, single-use `UnaryOp`s starting from `Fusion` inputs are handled via "input forwarding".
    -   **Tradeoff Analysis:** Saves intermediate tensor materialization and memory bandwidth at the cost of recomputing cheap ops.
    -   **Mechanism:** These unary expressions are initially excluded. Their final output is treated as a temporary "forwarded input". The excluded unary expressions are duplicated and prepended to each final consuming segment (`resolveForwardedInputs`).
    -   **Asymmetry:** Not applied for unary ops producing `Fusion` outputs.
-   **Up-Cast Privatization:** Up-casts (`CastOp` low-to-high precision) with multiple consumers are temporarily replicated (`privatizeUpcast`) to encourage merging on the lower-precision side, reducing memory bandwidth if a boundary occurs there. Duplicates are removed if they end up in the same final segment (`revertPrivatizedUpcast`).
-   **Other Unary Ops:** Placed in initial single-expression groups, available for merging.

#### Scalar Operations

-   **Initial Grouping & Edges:** Scalar-defining expressions are placed into their own `SegmentedGroup`s during `buildInitialSegments`. `SegmentedEdge`s representing scalar dependencies between groups are created and maintained throughout the process.
-   **Merging:** Scalar groups participate in the merging process just like tensor groups. Merges are validated by `tryMerge`, which calls `Schedule::proposeHeuristics`.
-   **Scheduling (`ExprEval`):** `Schedule::proposeHeuristics` identifies segments containing *only* scalar expressions and assigns them `SchedulerType::ExprEval`. This targets them for host execution using `ExpressionEvaluator`.
-   **Finalization (No Duplication):** During `SegmentedFusion::makeFusion`, the correct scalar `Val`s crossing the segment boundary are marked as inputs to the generated segment `Fusion`. Due to the Input/Output Boundary Principle, nvFuser ignores any definitions of these input scalars that might exist within the copied IR, preventing computation duplication.

#### Segmentation Hints (`SegmenterSet`)

-   The `LoadStoreOpType::SegmenterSet` operation acts as an explicit instruction to create a segment boundary.
-   Segments containing this operation are typically assigned `SchedulerType::ExprEval` by `Schedule::proposeHeuristics`.
-   This allows for integrating operations that cannot be directly code-generated into a GPU kernel (e.g., calls to external libraries or specific ATen functions) by executing them on the host via the `ExprEval` path. The surrounding computations can still be fused into GPU kernels.

### Precision at Boundaries

-   The segmenter attempts to place boundaries at lower precision (Half/BFloat16) when mixed-precision operations are involved, driven by the `IoToLowerPrecision` option.
-   **Mechanism:**
    -   `annotateFP16IntermediateTensors` identifies candidate FP32 intermediate tensors on paths to low-precision `Fusion` outputs.
    -   `castInputOutputToLowerPrecision` inserts `CastOp` pairs into the *complete `Fusion` IR* when such a tensor forms an edge between segments.
    -   The edge value (`SegmentedEdge::val`) is updated to the lower-precision tensor.
-   **Limitations:** Only applies to candidates based on final outputs; certain ops might prevent casting. FP32 boundaries can still occur.

-----

## Schedulability Check (`tryMerge` Mechanism)

-   **Purpose:** To ensure that any potential merge of `SegmentedGroup`s results in a new, larger group that can still be executed by a known mechanism.
-   **Process:**
    1.  The merging logic (`run_herrmann_merge`, `run_final_merge`, specific passes like `CombineReductions`) identifies candidate groups to merge.
    2.  Before committing to a merge, `codeGenSupportedMerge` (or similar logic) calls the `tryMerge` helper function.
    3.  `tryMerge` uses a `FusionSegmentGuard`. This guard temporarily modifies a *copy* or view of the `complete_fusion_` IR to represent the *potential* merged segment (setting its inputs and outputs).
    4.  Inside the guard, `tryMerge` calls `findScheduler`, which in turn calls `Schedule::proposeHeuristics` on the temporary, merged view of the fusion.
    5.  `Schedule::proposeHeuristics` analyzes the temporary fusion:
        -   It checks for GPU schedulability using heuristics like Pointwise, Reduction, Transpose, Normalization, Persistent.
        -   It *also* specifically checks if the fusion consists only of scalar operations or contains segmentation hints (`SegmenterSet`), in which case it returns `SchedulerType::ExprEval`.
        -   If no GPU heuristic applies and it's not an `ExprEval` case, it returns `SchedulerType::None`.
    6.  `tryMerge` returns the `SchedulerType` found by `proposeHeuristics`.
    7.  The calling merge logic proceeds with the merge *only if* the returned `SchedulerType` is *not* `SchedulerType::None`.
-   **Outcome:** This ensures that every merge maintains the invariant that the resulting group corresponds to a valid execution strategy (GPU kernel or host evaluation).

-----

## Future Optimizations / TODOs

-   **More Targeted `makeFusion` Cloning:** The current "minimal" `SegmentedFusion::makeFusion` implementation copies the entire `complete_fusion_` IR and then relies on correctly setting inputs/outputs to guide downstream processing (leveraging the Input/Output Boundary Principle). While correct, this full copy might be inefficient for very large original fusions. A future optimization could involve a more integrated traversal (e.g., forward or backward from outputs/inputs) to clone *only* the subgraph relevant to the segment (its expressions and boundary values) directly into a new `Fusion`, avoiding the large initial copy. This would achieve the same functional result but potentially improve segmenter performance.
-   **Detailed Merge Strategies & Heuristics Documentation:**
    -   Explain the goals and mechanisms of specific merging passes:
        -   `TranslateApplicableWelford` (Welford to 2-pass mean/var translation)
        -   `MergeUpAndDownCast` (Grouping cast sequences)
        -   `CombineReductions` (Merging compatible reduction/normalization ops)
        -   `run_herrmann_merge` (Main level-based merging, including `PreferredMergeCandidatePicker` logic)
        -   `run_final_merge` (Final cleanup merging pass)

## Current failing tests:
(previous failing tests section preserved as is) 