#include <ir_all_nodes.h>

// Return boolean predicates representing the conditional you want to assume.
// The return value is typically used as the `assumptions` argument of
// `simplifyExpr`

namespace nvfuser::assume {

// Return a boolean predicate stating that all tensor sizes appearing in `value`
// are positive. Return nullptr if `value` does not depend on any tensor size.
// For example:
//   tensorsAreNotEmpty(ceilDiv(T0.size[0], 5) * T0.size[1])
//     -> T0.size[0] > 0 && T0.size[1] > 0
//   tensorsAreNotEmpty(ceilDiv(i1, 5) * i2)
//     -> nullptr
Bool* tensorsAreNotEmpty(Val* value);

} // namespace nvfuser::assume
