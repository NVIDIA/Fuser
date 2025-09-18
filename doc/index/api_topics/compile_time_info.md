# Compile-time Info & HeuristicDataCache

## Synopsis
Cache of compile-time-derived facts used by schedulers to reduce dynamic-shape latency.

## Sources
- Entry enum: [`CompileTimeEntryType`](../../../csrc/scheduler/compile_time_info.h#L33)
- Data cache: [`HeuristicDataCache`](../../../csrc/scheduler/compile_time_info.h#L249)
- Access helper: `HeuristicDataCacheEntry<T>` (template) [`#L295`]

## Overview
Schedulers can precompute and cache facts such as domain maps, vectorizable IOs, persistent buffers, and more. The cache is keyed by `CompileTimeEntryType` and stores polymorphic entries. A convenience wrapper `HeuristicDataCacheEntry<EntryClass>` computes-or-loads entries and exposes `get()` to retrieve the typed data.

Common entries include:
- Domain maps: `DOMAIN_MAP`, `TRANSPOSE_DOMAIN_MAP`
- Reference tensors: `REFERENCE_TENSORS`, `REFERENCE_TENSORS_FOR_GROUPS`
- Vectorization and unrolling: `VECTORIZABLE_INPUTS_AND_OUTPUTS`, `UNROLLABLE_INPUTS_AND_OUTPUTS`, `TV_TO_CONTIG_INNER_SIZE_MAPS`
- Reduction and persistence: `REDUCTION_TVS`, `PERSISTENT_BUFFER_INFO`, `SCOPE_PERSISTENT_FACTOR_INFO`
- Broadcast multiples and inner-most dims info
- Flags and params: `CAN_SCHEDULE_TRANSPOSE`, `SCHEDULE_HYPERPARAMETERS`

## Example (pattern)
```cpp
HeuristicDataCache* cache = ...;
auto entry = HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensors>(
    cache,
    [&](){ return std::make_unique<std::vector<TensorView*>>(compute_refs()); });
auto& refs = entry.get();
```

## Related
- `scheduler/tools/domain_map.*`
- `scheduler/utils.*`
