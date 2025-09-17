# AdjustLastDim

Source: [AdjustLastDim](../../../csrc/type.h#L964)

## Synopsis
- **Kind**: struct
- **File**: `csrc/type.h`
- **Approx. size**: ~34 lines

## Context (from code comments)
NVFuser's DataType is much wider than PyTorch's ScalarType, and we do support
input/output TensorViews with these data types not supported by PyTorch.
For these cases, we use a PyTorch ScalarType as a proxy. If there exists
a scalar type with the same size, we use that. Otherwise, we use Byte and
and adjust the size of the last dimension. For example, if we have a
TensorView with shape [10, 4], and dtype is 3 bytes, then the corresponding
ScalarType is Byte, and the shape of the corresponding at::Tensor is [10,
12].

## Purpose
- Utility or analysis type contributing to scheduling/lowering.
