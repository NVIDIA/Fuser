#include <assume.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>

#include <vector>

namespace nvfuser::assume {

Bool* tensorsAreNotEmpty(Val* value) {
  std::vector<Val*> todo{value};
  std::vector<Val*> tensor_sizes;
  while (!todo.empty()) {
    auto v = todo.back();
    todo.pop_back();
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
  for (auto ts : tensor_sizes) {
    result = SimplifyingIrBuilder::andExpr(
        result, SimplifyingIrBuilder::gtExpr(ts, ts->container()->zeroVal()));
  }
  return result;
}

} // namespace nvfuser::assume