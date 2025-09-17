# IrContainer

Source: [IrContainer](../../csrc/ir/container.h#L35)

## Synopsis
- **Kind**: class (inherits from `PolymorphicBase`)
- **File**: `csrc/ir/container.h`
- **What it represents**: Abstract owner for IR graphs (values and expressions). [Fusion](../../csrc/fusion.h#L134) derives from `IrContainer`.

## Purpose
- Provides storage and lifecycle management for IR nodes, along with virtual hooks for adding/removing expressions and values.
- Enables cloning, visitation, and container-level operations independent of a specific fusion instance.

## See also
- [Fusion](../../csrc/fusion.h#L134), [Statement](../../csrc/ir/base_nodes.h#L96)
