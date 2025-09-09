# Dump and Inspect IR: Prints and Getters

## Key headers
- [`Fusion`](../csrc/fusion.h), [`FusionGuard`](../csrc/fusion_guard.h)
- [`TensorView`](../csrc/ir/interface_nodes.h)
- [`FusionExecutorCache`](../csrc/runtime/fusion_executor_cache.h)

## Tools
- `tv->printTransforms()` — per-TensorView transform history (root/logical/loop)
- `fusion.print(std::cout, /*include_tensor_transforms=*/true|false)` — with/without transforms
- `FusionExecutorCache` getters (after an execution):
  - `getMostRecentScheduledIr(/*tensor_transforms=*/true|false)`
  - `getMostRecentCode()` (generated CUDA)

References:
- Notes: `../doc-bot/experimenting/9-3-2025/key_insights.md`

---

## Example: static prints (no execution required)

```cpp
// After building IR
A->printTransforms();
B->printTransforms();
C->printTransforms();

fusion.print(std::cout, /*include_tensor_transforms=*/true);
fusion.print(std::cout, /*include_tensor_transforms=*/false);
```

- With transforms: includes per-TV domain sections and TransformPrinter
- Without transforms: concise op graph

---

## Example: runtime getters (require execution at least once)

```cpp
FusionExecutorCache fec(std::move(fusion_ptr));
auto outs = fec.runFusionWithInputs(args); // schedules and executes at least once

std::string sched_ir = fec.getMostRecentScheduledIr(/*tensor_transforms=*/true);
std::string cuda     = fec.getMostRecentCode();

std::cout << "=== Scheduled IR ===\n" << sched_ir << "\n";
std::cout << "=== Generated CUDA ===\n" << cuda << "\n";
```

Caveat:
- "Most recent" getters are populated only after scheduling/execution has occurred
- For pre-run inspection, rely on `printTransforms()` and `fusion.print(...)`

---

## Tips
- Prefer text dumps in documentation; avoid binary logs
- Use environment toggles (e.g., `NVFUSER_DUMP=cuda_to_file`) for deeper tracing when needed
- Combine IR printing with small, focused samples so readers can map constructs to output

See also:
- `./how-to-read-fusion-ir-dumps.md`
- `./fusion-ir-anatomy-and-printer.md`
