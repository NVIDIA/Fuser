## Host IR JIT vs CUDA Kernel JIT: When and Why

Prefer Host IR JIT when host-side orchestration latency dominates and supported host ops can be lowered efficiently; rely on LLVM ORC JIT and ATen wrappers.

CUDA kernel JIT is the standard path for GPU execution.

Refs: Q&A Topic 8 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.


