import Lake
open Lake DSL

package nvfuser where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib math.logic where

@[default_target]
lean_lib math.monotonic_function where

@[default_target]
lean_lib iterdomain where
