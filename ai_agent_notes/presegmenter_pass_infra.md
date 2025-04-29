# Pre-Segmentation Pass Infrastructure Notes

This document describes the C++ infrastructure used for defining and running optimization passes before the main fusion segmentation process in nvFuser.

## Core Components

The infrastructure revolves around a few key classes:

1.  **`nvfuser::Pass` (Base Class):**
    *   **Location:** Likely defined in `ir/base_nodes.h` or a similar core IR header.
    *   **Purpose:** Provides the fundamental interface for any operation that transforms a `Fusion`.
    *   **Key API (Assumed):**
        *   `Pass(Fusion* fusion)`: Constructor, potentially storing the fusion context.
        *   `virtual void runPass(Fusion* fusion) = 0;`: Pure virtual method that derived passes must implement to contain their transformation logic.
        *   `bool isModified() const;`: Returns whether the pass modified the `Fusion`. Passes are responsible for setting an internal `modified_` flag.
        *   `Fusion* fusion() const;`: Returns the associated fusion object.

2.  **`nvfuser::preseg_passes::OptimizationPass<DerivedClass>` (Template Manager):**
    *   **Location:** `csrc/preseg_passes/optimization_pass.h`
    *   **Purpose:** Acts as a manager or wrapper for specific optimization passes or sequences of passes. It provides a common interface for enabling/disabling passes and running them with standardized logging and potential restart logic. It uses the Curiously Recurring Template Pattern (CRTP).
    *   **Key API:**
        *   `static void setEnabled(bool enabled)`: Globally enables or disables the specific optimization pass represented by `DerivedClass`. Uses a static atomic flag (`flag_`).
        *   `static bool getEnabled()`: Checks if the pass type is currently enabled.
        *   `static void runPass(Fusion* fusion)`: The primary entry point.
            -   Checks if the pass type is enabled via `flag_`.
            -   Adds performance scoping (`FUSER_PERF_SCOPE`) using `DerivedClass::name()`.
            -   Provides optional debug logging (`DebugDumpOption::PreSegmenterLogging`) before and after the pass runs.
            -   Calls the static `DerivedClass::runPass(fusion)` to execute the actual logic (see note below).
        *   **Note on `runPass` Implementation:** The static `runPass` within `OptimizationPass` often contains a loop structure. It retrieves a list of individual `Pass*` instances by calling `DerivedClass::registerPasses()`. It then iterates through these registered passes, calling `pass->runPass(fusion)` on each. If any pass sets its `modified_` flag, the entire sequence of registered passes is **restarted** from the beginning in the next iteration of the `while(modified)` loop. This continues until a full iteration completes without any pass modifying the fusion.
        *   `static std::string name()`: `DerivedClass` is expected to provide a static `name()` method returning the pass name for logging.
        *   `static std::vector<Pass*> registerPasses()`: `DerivedClass` is expected to provide this static method, which returns the sequence of concrete `Pass` objects to be executed when `OptimizationPass<DerivedClass>::runPass` is called.

3.  **`nvfuser::preseg_passes::OptimizationPassGuard<OptPass>` (RAII Guard):**
    *   **Location:** `csrc/preseg_passes/optimization_pass.h`
    *   **Purpose:** Allows temporarily enabling or disabling a specific `OptimizationPass` type within a C++ scope.
    *   **Mechanism:** Stores the previous enabled state of `OptPass` on construction and restores it on destruction.

4.  **Concrete Pass Implementations (e.g., `MovePadPass`, `ConsecutiveCastPass`):**
    *   **Location:** Typically in `csrc/preseg_passes/` directory (e.g., `move_pad.cpp`).
    *   **Inheritance:** Usually inherit directly from `nvfuser::Pass`.
    *   **Implementation:**
        *   Provide a constructor `MyPass(Fusion* fusion) : Pass(fusion) {}`.
        *   Implement the core logic within `void runPass(Fusion* fusion) override;`.
        *   Crucially, set the `modified_` flag (inherited from `Pass`) to `true` if the fusion is altered in any way by the pass. This is essential for the restart logic in `OptimizationPass`.

5.  **Orchestrator Passes (e.g., `PreSegmenter`):**
    *   **Location:** `csrc/preseg_passes/pre_segmenter.cpp` and `.h`.
    *   **Inheritance:** Inherits from `OptimizationPass<PreSegmenter>` (using CRTP).
    *   **Purpose:** Defines and executes a specific, ordered sequence of optimization passes.
    *   **Implementation:**
        *   Provides the static `name()` method required by `OptimizationPass` (for logging/scoping).
        *   Provides a static `runPass(Fusion* fusion)` method.
        *   **Crucially, this `runPass` method *does not* typically rely on `registerPasses()`. Instead, it directly calls the static `OptimizationPass<ConcretePassType>::runPass(fusion)` method for each desired sub-pass in a hardcoded sequence.** This gives `PreSegmenter` full control over the order and conditional execution (like the `if (isOptionDisabled(DisableOption::ResizeScheduler))` check for `MovePadPass`).

## Execution Flow Example (`PreSegmenter`)

1.  Code calls `OptimizationPass<PreSegmenter>::runPass(my_fusion)` (or sometimes directly `PreSegmenter::runPass(my_fusion)`).
2.  The static `runPass` in `OptimizationPass<PreSegmenter>` (if called via the template) OR the direct `PreSegmenter::runPass` implementation is executed.
3.  This `runPass` function then executes a **fixed sequence** of calls:
    *   `OptimizationPass<RemoveEmptyPass>::runPass(my_fusion);` (This internally checks if `RemoveEmptyPass` is enabled and runs its logic, potentially looping if modifications occur *within* `RemoveEmptyPass` if it were registered that way - though simple passes usually don't modify).
    *   `OptimizationPass<TranslateRepeatToExpand>::runPass(my_fusion);`
    *   ...
    *   `if (isOptionDisabled(DisableOption::ResizeScheduler)) { OptimizationPass<MovePadPass>::runPass(fusion); }`
    *   ...
4.  There is **no** automatic restart loop *at the `PreSegmenter` level* based on modifications made by the sub-passes. The sequence defined in `PreSegmenter::runPass` runs exactly once.

This infrastructure allows for modular pass design. `OptimizationPass<T>` provides enable/disable flags and common entry points, while orchestrators like `PreSegmenter` define the specific order and control flow for executing sequences of passes. 