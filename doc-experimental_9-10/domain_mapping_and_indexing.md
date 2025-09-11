## Domain Mapping Across Producers/Consumers and Indexing

How axes align between producer and consumer TVs determines valid indexing and compute-at behavior. This guide covers the mapping machinery and where indexing is computed.

### Logical mapping and replay

- `LogicalDomainMap`: builds axis correspondences between TVs
- Best-effort replays align loop structures:

```1011:1024:/opt/pytorch/nvfuser/csrc/transform_iter.cpp
auto consumer_replay = BestEffortReplay::replayCasP(...);
consumer_replay.addComplimentLeafIDs(...);
```

### Indexing

- Index expressions are computed during lowering by indexing components (see `IndexCompute` and helpers)
- Producer indices are offset when consumer slices (e.g., `A[i+16]`), avoiding materialized sub-tensors

Q&A reference (Topic 15): Consumer slice uses offsets; no temporary extract tensors required.

### Practical notes

- Ensure transformed axes remain replayable; invalid merges/swizzles can break indexing.
- Keep consumer loop domain authoritative for scheduling; map producers accordingly.

### References

- Q&A: `../doc-bot/experimenting/8-28-2025/q_and_a.md` (Topic 2, 15)
- Transforms/replay: `../csrc/transform_iter.cpp`
- Indexing: `../csrc/index_compute.cpp`


