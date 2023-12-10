#include <polymorphic_value.h>
#include <serde/expr_utils.h>

namespace nvfuser::serde {

std::vector<nvfuser::Val*> gatherSymbolicValues(kir::Kernel* kernel) {
  std::vector<nvfuser::Val*> symbolic_values;
  for (auto input : kernel->inputs()) {
    if (auto tv = dynamic_cast<nvfuser::TensorView*>(input)) {
      insertUniqueItem(symbolic_values, tv);
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        if (!extent->isA<nvfuser::NamedScalar>() && !extent->isConstInt()) {
          insertUniqueItem(symbolic_values, extent);
        }
      }
    }
  }
  return symbolic_values;
}

} // namespace nvfuser::serde
