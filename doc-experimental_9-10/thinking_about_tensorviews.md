## How to Think About TensorViews (Symbolic View, Not Data)

TensorView is a symbolic handle to domains and transforms; it carries no storage. Data arrives at execution time via `at::Tensor` bindings.

Key points:
- Build IR with `TensorViewBuilder`; register inputs/outputs on `Fusion`
- Use transforms (reshape/squeeze/broadcast/permute) to alter logical domains
- Execution binds runtime `at::Tensor`s and selects a schedule; TensorView remains symbolic

Reference: `../doc-bot/experimenting/9-3-2025/key_insights.md` and `../csrc/ops/alias.cpp`.


