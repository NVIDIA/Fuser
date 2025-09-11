## Dumping IR and Code: printFusion, printTransforms, getMostRecent*

During development, print Fusion IR and transforms, and after execution retrieve scheduled IR and CUDA code.

```36:46:/opt/pytorch/nvfuser/doc-bot/experimenting/8-26-2025/tv_deeper_dive.cpp
fusion->print(std::cout, /*include_tensor_transforms=*/true);
fusion->print(std::cout, /*include_tensor_transforms=*/false);
```

```157:176:/opt/pytorch/nvfuser/csrc/runtime/fusion_executor_cache.h
// getMostRecentCode(), getMostRecentScheduledIr(), getScheduledIrFor(...)
```

Tip: The “most recent” getters are valid after at least one execution.


