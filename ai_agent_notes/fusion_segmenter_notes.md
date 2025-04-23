# NVFuser Fusion Segmenter Development Notes

## Goals and Motivation

-   **Why Segment?** Fusing an entire complex computation into a single GPU kernel is often impractical or suboptimal due to:
    -   Operations unsupported by the GPU kernel code generator (e.g., certain communications, host-side logic).
    -   Complex control flow or graph structures that are difficult to schedule efficiently as one unit.
    -   Different parts of the graph having conflicting optimal scheduling strategies (e.g., reductions vs. pointwise ops).
    -   Excessive register pressure or shared memory usage in a monolithic kernel.
-   **Segmentation Addresses This By:** Breaking the computation graph into smaller, manageable sub-graphs (`SegmentedGroup`s). Each segment can then be independently scheduled and potentially compiled into a separate, optimized kernel or executed on the host.

## Core Principle: Schedulability-Driven Partitioning

-   Segmentation is not arbitrary graph cutting. The primary constraint is that **each final `SegmentedGroup` must be schedulable** by a known heuristic (e.g., Pointwise, Reduction, Persistent).
-   The merging process (`SegmentCandidateFinder`) iteratively combines smaller groups, but crucially **validates each potential merge** by checking if the resulting larger group can still be scheduled (`tryMerge` mechanism using `Schedule::proposeHeuristics`).
-   The fundamental output is a partitioning of the original fusion into a DAG of **schedulable** execution units (GPU kernels or potential host sections).

## Key Performance Tradeoffs

Segmentation involves balancing several competing performance factors:

-   **Kernel Launch Overhead:** More segments generally mean more distinct kernels to launch, adding CPU overhead.
-   **Memory Bandwidth (Inter-Segment):** Segment boundaries usually imply intermediate tensor results being written to and read from global memory. Fewer boundaries often reduce this I/O, which is frequently a bottleneck.
-   **Scheduling Efficiency:** Smaller, more specialized segments might allow for more optimal scheduling heuristics tailored to that segment's operations (e.g., highly optimized reduction kernels), potentially leading to faster execution *within* the segment compared to a less specialized schedule for a larger, combined segment.
-   **Recomputation vs. Memory I/O:** Sometimes, recomputing cheap operations (like simple unary ops via Input Forwarding) within multiple subsequent segments is faster than paying the memory cost to materialize and transfer the intermediate result once.
-   **Potential Tradeoff (Dimensionality - Currently Less Explicitly Optimized):** The size of data transferred at segment boundaries can be affected by dimensionality changes. Ideally, boundaries would occur *after* operations that reduce data size (like reductions) and *before* operations that increase data size (like broadcasts) to minimize memory traffic. While not the primary driver now, this influences the benefit of strategies like `CombineReductions`.

## Conceptual Configuration

-   The segmentation process includes several optional passes or strategies controlled by `SegmentCandidateFinderOptions`.
-   These passes (e.g., `run_combine_reductions`, `run_translate_welford`, `run_merge_up_and_down_cast`) apply specific heuristics to merge groups based on known beneficial patterns *before* or *during* the main merging loop, influencing the final segmentation outcome.

## High-Level Workflow Summary

The `SegmentCandidateFinder` class orchestrates the segmentation process:

1.  **Input:** Takes an original `Fusion` DAG.
2.  **Initial Segmentation (`buildInitialSegments`):** Traverses the `Fusion`'s expressions. Creates fine-grained `SegmentedGroup`s, typically one non-scalar operation per group. Scalar operations are initially ignored.
3.  **Input/Scalar Handling Prep:** Applies strategies like Input Forwarding (for cheap unary ops near inputs) and Up-Cast Privatization to prepare the graph for merging. Removes edges based on scalar values.
4.  **Optional Pre-Merge Passes:** Executes specialized merging passes based on `SegmentCandidateFinderOptions` if enabled (e.g., `TranslateApplicableWelford`, `MergeUpAndDownCast`, `CombineReductions`). These passes group operations based on specific known patterns (like reductions, casts).
5.  **Iterative Merging (`run_herrmann_merge`, `run_final_merge`):** The core merging loop. Iteratively identifies candidate pairs or sets of adjacent `SegmentedGroup`s to merge based on heuristics (like DAG levels, operation types). **Crucially, each potential merge is validated using `tryMerge` to ensure the resulting combined group is schedulable by a known scheduler heuristic.** Merging proceeds only if schedulable.
6.  **Finalization (`finalize`, `resolveForwardedInputs`, `resolveScalarsInGroup`):** Cleans up the graph. Resolves previously forwarded inputs by prepending their defining unary ops to the consuming segments. Resolves scalar dependencies by adding necessary scalar computations into each segment. Finalizes segment inputs/outputs.
7.  **Output:** Produces a `SegmentedFusion` object containing the final DAG of schedulable `SegmentedGroup`s, ready for execution scheduling and code generation.

## High-Level Summaries (TLDRs)

-   **TLDR: Edge System:** Manages data dependencies between segments using `SegmentedEdge` objects and centralized helper functions. Ensures correct graph structure and data flow, especially during segment merging. *(Details below)*
-   **TLDR: Handling Specific Ops:** Special logic exists for certain operations. Cheap unary ops near inputs are often recomputed ("forwarded") to save memory I/O. Up-casts are handled to encourage lower-precision boundaries. Scalar ops are initially ignored during merging and their logic is localized into the final GPU segments (though future work aims to leverage host execution). *(Details below)*
-   **TLDR: Precision at Boundaries:** When enabled (`IoToLowerPrecision`), the segmenter attempts to insert casts such that intermediate tensors involved in mixed-precision paths are transferred between segments in lower precision (Half/BFloat16), reducing memory bandwidth. This is targeted based on final fusion outputs and subject to operation constraints. *(Details below)*

-----

## Detailed Explanations

### Edge System: Tracking Dependencies

-   **Purpose:** The edge system is crucial for representing and maintaining data dependencies between `SegmentedGroup`s throughout the segmentation process. It ensures that as groups are merged, the correct data flow is preserved.
-   **Representation:**
    -   Dependencies are modeled using `SegmentedEdge` objects. Each edge connects a producer group (`from`) to a consumer group (`to`) and specifies the exact intermediate `Val` being passed between them.
    -   Multiple edges can exist between the same two groups if multiple distinct `Val`s are passed.
    -   Each `SegmentedGroup` maintains lists of its incoming (`producer_edges`) and outgoing (`consumer_edges`) connections.
-   **Role in Merging:**
    -   Edges are used to identify neighboring groups and potential merge candidates.
    -   When groups are merged, the edge system logic determines which external edges need to be reconnected to the new, merged group. Edges *internal* to the merged set are discarded.
    -   Centralized helper functions (like `connectGroups`, `removeEdge`, `getEdgesBetween`) are used internally by the segmenter to ensure these modifications are done consistently and correctly, preserving the DAG structure.
-   **Data Structures:** Edges are managed centrally within the `SegmentedFusion` object, while groups hold references to their relevant producer/consumer edges. Input/output `Val`s within groups use `VectorOfUniqueEntries` for efficient handling.

### Handling of Specific Operation Types

#### Unary Operations

-   **Input Forwarding:** Chains of simple, single-use `UnaryOp`s starting from fusion inputs are handled via "input forwarding". This addresses the scenario: `Input -> Unary Ops -> N Consumers`.
    -   **Tradeoff Analysis:** Compares recomputing the unary ops vs. materializing their intermediate result.
        1.  *Materializing Intermediate (Separate Unary Kernel or Fused with One Consumer):* Requires 1 Read (`Input`) + (N or N-1) Reads (`Intermediate`) + 1 Write (`Intermediate`).
        2.  *Recomputing Unary (Input Forwarding):* Requires N Reads (`Input`) + 0 Writes (`Intermediate`).
        3.  *Benefit of Recomputation:* Saves 1 Write and N (or N-1) Reads of the `Intermediate` tensor's data from global memory, at the cost of N-1 extra Reads of the `Input` tensor's data. For cheap unary ops, this significantly reduces memory bandwidth usage.
    -   **Mechanism:** These unary expressions are initially excluded from the segmentation graph. The final output of the chain is treated as a temporary "forwarded input" during merging. The excluded unary expressions are duplicated and prepended to each final `SegmentedGroup` that uses the corresponding forwarded input (`resolveForwardedInputs`).
    -   **Asymmetry:** This forwarding logic is *not* applied similarly for cheap unary ops producing fusion *outputs*, primarily because the main benefit (avoiding intermediate global memory traffic) is less significant, as the final output usually needs to be written anyway. The `shouldForward` function explicitly prevents forwarding if the unary op produces a fusion output.
-   **Up-Cast Privatization:** To influence merging behavior around data type changes:
    -   Up-cast operations (`CastOp` from lower to higher precision) with multiple consumers are temporarily replicated (`privatizeUpcast`). Each use gets its "own" cast op initially.
    -   This encourages merging on the lower-precision side of the cast. **This is beneficial because if a segment boundary must exist near the cast, placing it *before* the up-cast means the smaller, lower-precision tensor is transferred via global memory, reducing bandwidth usage compared to transferring the larger, higher-precision tensor.**
    -   If multiple privatized versions of the same original up-cast end up in the same final segment, the duplicates are removed (`revertPrivatizedUpcast`).
-   **Other Unary Ops:** Unary operations not part of input forwarding or up-cast privatization are initially placed in their own single-expression `SegmentedGroup`. **This provides maximum granularity for the subsequent heuristic-driven merging passes.**

#### Scalar Operations

-   **Initial Exclusion:** Scalar operations are generally ignored during the main graph building and merging phases.
    -   `SegmentedGroup`s are not created for scalar-only expressions in `buildInitialSegments`.
    -   Edges based on scalar `Val`s are explicitly removed (`removeScalarEdges`) to prevent them from influencing merging decisions, as scalar dependencies typically don't require kernel boundaries.
-   **Late Resolution (Current Strategy):** Scalar computations are localized within the final GPU kernel segments that need them.
    -   **Motivation (Historical):** This approach was necessary because nvFuser historically lacked a host-based executor capable of computing scalar values and passing them into kernels or returning them as distinct fusion outputs. Pushing all scalar logic onto the GPU was the only option, although potentially inefficient due to redundant computation across GPU threads.
    -   **Mechanism:**
        -   After merging, `resolveScalarsInGroup` analyzes each `SegmentedGroup`.
        -   It identifies all required scalar values. This includes scalars used directly by expressions, scalar attributes, and scalars needed to define the shapes of intermediate tensors computed within the group.
        -   **Tensor Shape Handling:** It recognizes that shape information (extents) of segment *input* tensors is implicitly available to the kernel. If such a shape value is used in a subsequent scalar computation within the segment, the trace recognizes it as available and doesn't attempt to add expressions to recompute it.
        -   It performs a backward dependency traversal starting from required scalars (excluding implicitly available input shapes), using a `visited` set to handle cycles/redundancy.
        -   The traversal finds the defining expressions for computed intermediate scalars and recursively adds their scalar inputs to the set of required scalars, stopping at constants or fusion inputs.
        -   The necessary scalar-defining expressions encountered during the trace are collected and added to the `exprs_` list of the `SegmentedGroup`.
        -   `SegmentedGroup::finalize` also contributes to ensuring scalar dependencies are correctly included.
-   **New Capability & Chosen Future Direction (Approach 4 - Heuristic Modification):**
    -   nvFuser now possesses host-based scalar execution capabilities (e.g., via `ExpressionEvaluator`, potentially used by executors like `HostIrEvaluator`).
    -   The plan is to modify segmentation heuristics (`codeGenSupportedMerge`/`deriveSchedulerType`) to leverage this.
    -   **Goal:** Identify segments dominated by scalar computations that are suitable for host execution. These segments could be assigned a special "Host" execution type, distinct from GPU kernel types (`SchedulerType::...`).
    -   This aims to reduce redundant scalar computation previously pushed to the GPU and potentially enable direct scalar outputs from fusions.
    -   **(Alternative Idea - Approach 2):** An alternative worth considering separately would be pre-segmentation extraction of scalar blocks.
    <!-- TODO: Discuss scalar resolution process further, potentially in context of host execution -->

### Precision at Boundaries

- The segmenter attempts to place boundaries at lower precision (Half/BFloat16) when mixed-precision operations are involved, driven by the `IoToLowerPrecision` option.
- **Mechanism:**
    - `annotateFP16IntermediateTensors` identifies candidate FP32 intermediate tensors that are on a path leading to a Half/BFloat16 *output* of the complete fusion.
    - `castInputOutputToLowerPrecision` inserts `CastOp` pairs (e.g., FP32->Half, Half->FP32) into the *complete fusion IR* when a candidate tensor forms an edge between segments.
    - The edge value (`SegmentedEdge::val`) is updated to the lower-precision tensor, while the consumer segment uses the output of the cast-back op.
- **Limitations:** This is not a universal rule. It only applies to candidate tensors identified based on final outputs, and certain consuming operations might prevent the cast insertion. FP32 boundaries can still occur.

-----

## Current failing tests:
(previous failing tests section preserved as is)

## TODO: Missing Documentation Sections

-   **Detailed Merge Strategies & Heuristics:**
    -   Explain the goals and mechanisms of specific merging passes:
        -   `TranslateApplicableWelford` (Welford to 2-pass mean/var translation)
        -   `MergeUpAndDownCast` (Grouping cast sequences)
        -   `CombineReductions` (Merging compatible reduction/normalization ops)
        -   `run_herrmann_merge` (Main level-based merging, including `PreferredMergeCandidatePicker` logic)
        -   `run_final_merge` (Final cleanup merging pass)
-   **Schedulability Check (`tryMerge` Mechanism):**
    -   Detail how `FusionSegmentGuard` works to temporarily modify the fusion for schedulability testing.
    -   Explain the role of `Schedule::proposeHeuristics` in validating potential merges. 