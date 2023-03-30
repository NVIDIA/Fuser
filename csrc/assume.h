#include <ir_all_nodes.h>

// Return boolean values representing the conditional you want to assume

namespace nvfuser::assume {

// Assume that all tensor sizes appearing in `value` are positive. Return
// nullptr if not applicable. For example:
//   tensorsAreNotEmpty(ceilDiv(T0.size[0], 5) * T0.size[1])
//     -> T0.size[0] > 0 && T0.size[1] > 0
//   tensorsAreNotEmpty(ceilDiv(i1, 5) * i2)
//     -> nullptr
Bool* tensorsAreNotEmpty(Val* value);

} // namespace nvfuser::assume
