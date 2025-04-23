# Notes on Segmentation Changes for Host Scalar Execution

**Objective:** Modify the fusion segmenter to identify and potentially segment purely scalar computations for execution via a host-based executor (likely `ExprEvalExecutor` or similar using `ExpressionEvaluator`).

**Summary of Changes Observed in `csrc/fusion_segmenter.cpp` and `csrc/fusion_segmenter.h` (compared to `main`):**

1.  **`deriveSchedulerType` / `codeGenSupportedMerge` Modifications:**
    *   Logic appears to have been added within `deriveSchedulerType` (or the `tryMerge` function it calls) to check if a `SegmentedGroup` consists *solely* of scalar operations (e.g., iterating through `group->exprs()` and checking `ir_utils::isScalarOp`).
    *   If a group is identified as purely scalar, the function seems modified to return `SchedulerType::ExprEval` (or a similar new/existing type indicating host execution) instead of `SchedulerType::None` or attempting GPU scheduling heuristics. *(Note: Initial analysis incorrectly assumed this check was unreachable due to merge validation failures; see below)*.

2.  **`buildInitialSegments`:**
    *   The logic that *skips* creating initial `SegmentedGroup`s for individual scalar expressions seems **unchanged**. Scalar ops are still not represented as initial, mergeable groups.

3.  **`resolveScalarsInGroup`:**
    *   This function, which pulls necessary scalar computations *into* final GPU segments, appears **unchanged**. It likely still runs for all GPU-targeted segments.

4.  **`removeScalarEdges`:**
    *   The call to remove edges based on scalar `Val`s after initial segmentation appears **unchanged**.

**Refined Understanding of Merge Validation (Resulting from Debug Discussion):**

*   The initial assessment suggested that `tryMerge` would fail for purely scalar merges because `proposeHeuristics` wouldn't find a GPU scheduler.
*   **Correction:** `Schedule::canSchedule` (called by `proposeHeuristics`) explicitly *allows* `SchedulerType::ExprEval` for fusions containing only scalar-producing expressions, while rejecting GPU types. Therefore, `proposeHeuristics` *should* return `SchedulerType::ExprEval` for such cases, and `tryMerge` should consequently *allow* the merge.
*   This refinement shifts the focus from merge validation failure to the finalization step as the primary bottleneck.

**Potential Issues/Observations based on Diff:**

*   **Initial Merging Path Now Possible:** While `buildInitialSegments` creates initial `SegmentedGroup`s for scalar ops, the key `Schedule::canSchedule` function (in `scheduler/registry.cpp`) explicitly *rejects* GPU schedulers for purely scalar groups but *accepts* `SchedulerType::ExprEval`. This means `proposeHeuristics`, and subsequently `tryMerge`, *could* now potentially allow merges resulting in purely scalar groups marked as `SchedulerType::ExprEval`.
*   **Redundant Scalar Computation / Finalization Conflict: [Confirmed Primary Issue]** Even if a purely scalar segment is formed and marked for `ExprEval`, the *primary remaining issue* is `resolveScalarsInGroup`. This function runs during finalization for **GPU-targeted segments**. If a GPU segment requires a scalar result produced by the `ExprEval` segment, `resolveScalarsInGroup` will still trace the dependency back and **duplicate the necessary scalar `Expr`s into the GPU segment's `exprs_` list**. This leads to redundant computation and negates the benefit of the separate host segment.
*   **Lack of Host->GPU Scalar Value Passing:** The current finalization logic does not seem equipped to handle passing a scalar value computed in a host segment (`ExprEval`) as a direct input (e.g., kernel argument) to a subsequent GPU segment. `resolveScalarsInGroup` effectively replaces this potential dependency path by duplicating the computation.

**Summary:** The changes likely allow `tryMerge` to *approve* merges that result in purely scalar groups intended for host execution (`SchedulerType::ExprEval`). However, the **fundamental incompatibility lies in the finalization step**: `resolveScalarsInGroup` still forces scalar logic duplication into consuming GPU kernels, preventing true separation and host-only execution of those scalars. The mechanism to pass computed host scalars to GPU kernels is missing.

## Next Steps / Proposed Fixes

Based on the identified issues, the following modifications are needed to enable host execution of scalar segments:

1.  **Modify Finalization Logic for GPU Segments:**
    *   **Problem:** `resolveScalarsInGroup` currently duplicates scalar `Expr`s from preceding host segments into GPU segments.
    *   **Proposed Fix:** Update `resolveScalarsInGroup` (or the overall finalization process for GPU segments) to recognize when a required scalar input is the output of a segment marked for host execution (`SchedulerType::ExprEval`). Instead of tracing back and duplicating the scalar `Expr`s, it should treat this scalar `Val` as a direct input to the GPU segment, similar to how tensor inputs from other segments are handled.

2.  **Implement Host-to-GPU Scalar Value Passing:**
    *   **Problem:** There's no mechanism to pass the actual scalar value computed by a host segment to a subsequent GPU kernel that requires it.
    *   **Proposed Fix:** Integrate with the runtime/executor system. When a GPU kernel segment follows a host segment, the execution flow must:
        a.  Execute the host segment (using `ExpressionEvaluator` or similar).
        b.  Retrieve the resulting scalar value(s).
        c.  Pass these scalar values as kernel arguments (likely alongside tensor pointers and sizes) when launching the subsequent GPU kernel. 