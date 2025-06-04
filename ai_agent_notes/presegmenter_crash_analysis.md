# Analysis of Pre-Segmenter Crash during Parallel Test Runs

Date: 2024-07-16

## Observation

When running the test suite with the filter `*Scheduler*` in parallel across 4 GPUs using the `run_multiple_times.sh` script, segfaults (SIGSEGV) or bus errors (SIGBUS) were observed intermittently on GPUs 1, 2, and 3. GPU 0 consistently completed without crashing.

The crashes consistently occurred during the execution of the `ResizeTest.SliceReduceScheduler2` test case.

Based on the added debug logging, the crash point was isolated to occur *during* the execution of the pre-segmenter passes, specifically within the call stack originating from:

```c++
// In FusionKernelRuntime::FusionKernelRuntime constructor
preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(fusion.get());
```

The crash happens after the `[RUNTIME CONSTRUCTOR] After NVF_ERROR` log and before the first `[PreSegmenter] Running ...` log message, indicating the failure is either in the setup of `OptimizationPass::runPass` or very early in the first pass executed by `PreSegmenter`.

*   **Initial Crash Point:** In the first iteration observed, the crash on GPUs 1, 2, and 3 consistently occurred *during* the execution of the `TranslateRepeatToExpand` pass (i.e., after `[PreSegmenter] Running TranslateRepeatToExpand...` but before `[PreSegmenter] Finished TranslateRepeatToExpand.`).
*   **Consistent Crash Point:** A second observation confirmed that the crash on GPUs 1, 2, and 3 again occurred during the `TranslateRepeatToExpand` pass. This strongly suggests the issue lies within this specific pass or its interaction with concurrent execution.
*   **Shifted Crash Point (Run 3):** After adding detailed logging within `TranslateRepeatToExpand`, the logs from GPU 1 (which crashed) showed that all pre-segmenter passes, including `TranslateRepeatToExpand`, completed successfully for the `ResizeTest.SliceReduceScheduler2` fusion. The crash occurred *after* the line `[RUNTIME CONSTRUCTOR] After preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass` but *before* the next major step logged (`[RUNTIME CONSTRUCTOR] Preparing runtime order.`). This pinpoints the issue to the transition between the pre-segmenter phase and the runtime preparation phase within the `FusionKernelRuntime` constructor.
*   **Increased Variability (Run 3):** In this run, GPU 3 *passed* the `ResizeTest.SliceReduceScheduler2` test. GPU 2 failed with an assertion in a different, earlier test (`ResizeSchedulerTest.PropagateMultipleSlicesToInputs6`) and did not reach the target test.

## Analysis

*   **Inconsistent Occurrence:** The crash does not happen on every GPU or every run, suggesting a race condition or memory corruption issue related to concurrency. The variability increased in the latest run.
*   **Non-Parallel Test Failure:** The specific test `ResizeTest.SliceReduceScheduler2` likely has only one segment, meaning it does *not* utilize the intra-fusion parallel compilation thread pool. However, the crash still occurs when the *global* parallel compilation setting is enabled.
*   **Inter-Process Interference:** Since tests run in separate processes for each GPU, direct shared memory between the tests is unlikely. However, the concurrency might be causing issues through:
    *   **Driver/Runtime Contention:** Concurrent interactions with the CUDA driver, NVRTC, or CUDA context management might be non-thread-safe or expose driver bugs.
    *   **Filesystem/Resource Contention:** Less likely, but contention for temporary files or other system resources could play a role.
    *   **Shared State:** Although processes are separate, there might be unforeseen shared state at a lower level (e.g., within the driver, system libraries, or potentially static initialization issues within nvFuser itself if not handled carefully across processes, though this is less common).
    *   **Initialization Order / Timing:** The act of enabling the parallel compile feature globally (even if not used by the specific crashing test) might alter the initialization order or timing of certain components, exposing a latent bug in the pre-segmenter passes or the resources they access.
*   **State Corruption:** The latest logs strongly suggest that the pre-segmenter passes themselves complete, but potentially leave the `Fusion` object or related state (e.g., memory managed by the `Fusion` object) in a corrupted state. This corruption leads to a crash when subsequent operations in the runtime constructor (like `prepareRuntimeOrder`) attempt to use this state.

## Next Steps

1.  **Instrument `FusionKernelRuntime` Constructor:** Add logging immediately *before* and *after* the call to `preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(fusion.get());`.
2.  **Instrument `prepareRuntimeOrder`:** Add logging at the very beginning of the `prepareRuntimeOrder` method within the `FusionKernelRuntime` constructor. This will help confirm if the crash occurs exactly between the pre-segmenter execution and the start of runtime preparation.

## Separate Issue

The tests `NVFuserTest.FusionMagicSchedulerLayerNormalization_CUDA` and `NVFuserTest.FusionMagicSchedulerRMSNormalization_CUDA` consistently fail on all GPUs with a scheduling logic error (`Could not schedule fusion with the SchedulerType: inner_persistent`). This appears unrelated to the segfault and needs separate investigation. 