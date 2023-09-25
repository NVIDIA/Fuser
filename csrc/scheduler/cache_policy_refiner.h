#pragma once

#include <fusion.h>

namespace nvfuser {

// Visits all global-to-local vector loads in `fusion` and refines their cache
// policies.
void refineCachePolicy(Fusion* fusion);

} // namespace nvfuser
