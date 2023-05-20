#include <assume.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>

#include <vector>

namespace nvfuser::assume {

Bool* tensorsAreNotEmpty(Val* value) {
  std::vector<Val*> todo{value};
  std::vector<Val*> tensor_sizes;
  while (!todo.empty()) {
    auto v = todo.back();
    todo.pop_back();
    TORCH_INTERNAL_ASSERT(v != nullptr);
    if (auto ns = dynamic_cast<NamedScalar*>(v)) {
      if (ns->isTensorSize()) {
        tensor_sizes.emplace_back(v);
        continue;
      }
    }
    if (auto def = v->definition()) {
      for (auto inp : def->inputs()) {
        todo.emplace_back(inp);
      }
    }
  }
  Bool* result = nullptr;
  // tensor_sizes might contain duplicate, and we should remove this duplication
  std::vector<Val*> tensor_sizes_applied;
  for (auto ts : tensor_sizes) {
    bool is_duplicate = false;
    for (auto existing : tensor_sizes_applied) {
      if (existing->sameAs(ts)) {
        is_duplicate = true;
        break;
      }
    }
    if (!is_duplicate) {
      tensor_sizes_applied.emplace_back(ts);
      result = SimplifyingIrBuilder::andExpr(
          result, SimplifyingIrBuilder::gtExpr(ts, ts->container()->zeroVal()));
    }
  }
  return result;
}

} // namespace nvfuser::assume
