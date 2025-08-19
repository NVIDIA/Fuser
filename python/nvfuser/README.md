<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# nvFuser Python Frontend

This frontend allows for a user to describe the set of operations for nvFuser to fuse via 1 or more kernels.  This frontend is intended to be an integration point with PyTorch or standalone applications.

# Usage

## Example 1 - Define and Execute a Fusion

```python
import torch
from nvfuser import FusionDefinition, DataType

with FusionDefinition() as fd :
    t0 = fd.define_tensor(shape=[-1, 1, -1],
                          contiguity=[True, None, True],
                          dtype=DataType.Float)
    t1 = fd.define_tensor([-1, -1, -1])
    c0 = fd.define_scalar(3.0)

    t2 = fd.ops.add(t0, t1)
    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

    fd.add_output(t4)

input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')

nvf_out = fd.execute([input1, input2])[0]
```

## Example 2 - Lookup and Execute a `FusionDefinition` Based on Id

<!-- CI IGNORE -->
```python
fid = 0
fd = FusionDefinition(fid)

input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')

nvf_out = fd.execute([input1, input2])[0]
```

## Components

### `FusionDefinition` Context Manager - Interface for Defining Fusions
* `execute([inputs])`:  Allows you to execute the currently defined fusion with a list of given inputs and returns a list of tensors.
* `id()`: Returns the fusion id for a given definition.
* `fusion_ir()`: Returns the Fusion IR (Intermediate Representation) as a string.
* `last_cuda_code(intrinsic_code=False)`: Returns the generated CUDA code for the last executed inputs.
* `debug_output()`: Returns debug output if capture_debug_output=True was used during execution.

#### Defining Input Tensors
_All intermediate tensors are created by operations.  Constant tensors do not exist._

There are 3 ways to define tensors that will be enumerated below.

##### 1.) Defining tensors with symbolic dimensions
This interface tells nvFuser that the tensor has symbolic dimensions that are not necessarily contiguous in memory. Use `-1` for each symbolic dimension. The user also has the ability to specify a data type. The default type is `Float`.
```python
t0 = fd.define_tensor([-1, -1, -1])                        # 3D tensor
t1 = fd.define_tensor([-1, -1], dtype=DataType.Half)       # 2D tensor
```

##### 2.) Defining tensors by a list of concrete sizes and a list of strides
The `sizes` parameter defines the number of dimensions and the size of each dimension.  The `strides` parameter has to have the same number of dimensions as the `sizes` parameter.
nvFuser translates the concrete sizes and strides into symbolic sizes and contiguity information that can be directly defined via the next way to define tensors.  This allows the user to directly take a Pytorch defined tensor and query its sizes and strides in order to apply them in the definition.
```python
t0 = fd.define_tensor(sizes=[2, 4, 6], strides=[24, 6, 1], dtype=DataType.Half)
```

##### 3.) Defining tensors by a list of symbolic sizes and a list of contiguity information
The list of symbolic sizes defines the number of dimensions and `-1` is given for each dimension unless it is a broadcast dimension that is defined with a `1`.  The contiguity information is viewed from right to left.  A `True` definition indicates the current dimension is contiguous with the dimension to its right.

```python
t0 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True], dtype=DataType.Float)
```

#### Defining Input Scalars
_All intermediate scalars, except for constants, are created by operations._

The only thing the user has to define for a scalar is its type.

```python
s0 = fd.define_scalar(dtype=DataType.Half)
```

#### Defining Constant Scalars

Constants can be of types: `Bool`, `ComplexDouble`, `Double`, or `Int`.  The definition only takes a constant and the type is inferred by the constant's type.

```python
c0 = fd.define_scalar(3.0)
```

**Note**: you cannot use Python literals directly:
<!-- CI IGNORE -->
```python
# Correct - define scalar constant first
scalar_const = fd.define_scalar(2.0)
result = fd.ops.mul(tensor, scalar_const)

# Incorrect - this will cause a TypeError
result = fd.ops.mul(tensor, 2.0)  # ERROR!
```

#### Defining Operations

Operators are added with the following notation:
<!-- CI IGNORE -->
```python
output = fd.ops.foo(arg1, ... )
```


You can see a supported list of operations with the following query:
```bash
python -c "from nvfuser import FusionDefinition; help(FusionDefinition.Operators)"
```

#### Notating Outputs

The `FusionDefinition` `add_output` method is used to indicate an intermediate is an output to the fusion. Output can be a tensor or a scalar.

```python
t0 = fd.define_tensor(sizes=[2, 4, 6], strides=[24, 6, 1], dtype=DataType.Half)
fd.add_output(t0)
```
or
<!-- CI IGNORE -->
```python
fd.add_output(output: Scalar)
```

# Complete Working Example

Here's a complete, tested example that demonstrates correct API usage:

```python
import torch
from nvfuser import FusionDefinition, DataType

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. nvfuser requires CUDA.")
        return

    # Define a fusion that computes (x + y) * 2
    with FusionDefinition() as fd:
        # Define input tensors with explicit shapes
        x = fd.define_tensor([-1, -1], dtype=DataType.Float)  # 2D tensor
        y = fd.define_tensor([-1, -1], dtype=DataType.Float)  # 2D tensor

        # Define operations
        sum_result = fd.ops.add(x, y)       # x + y
        two = fd.define_scalar(2.0)         # scalar constant
        final_result = fd.ops.mul(sum_result, two)  # (x + y) * 2

        # Mark output
        fd.add_output(final_result)

    # Create input tensors on GPU
    input_x = torch.ones(3, 4, device='cuda', dtype=torch.float32)
    input_y = torch.ones(3, 4, device='cuda', dtype=torch.float32) * 2

    # Execute the fusion
    nvf_result = fd.execute([input_x, input_y])[0]

    # Compare with PyTorch eager execution
    eager_result = (input_x + input_y) * 2.0

    print(f"Results match: {torch.allclose(nvf_result, eager_result)}")

    # Get debug information (only available after execution)
    print(f"Fusion ID: {fd.id()}")
    print(f"Fusion IR:\n{fd.fusion_ir()}")

if __name__ == "__main__":
    main()
```

# Debug Information
**Query a list of supported operations:**
```bash
python -c "from nvfuser import FusionDefinition; help(FusionDefinition.Operators)"
```

**Get debug information after execution:**
<!-- CI IGNORE -->
```python
# These methods require the fusion to be executed first
print(f"Fusion ID: {fd.id()}")
print(f"Fusion IR:\n{fd.fusion_ir()}")
print(f"Generated CUDA code:\n{fd.last_cuda_code()}")
```

**View the fusion definitions that are executed by setting an environment variable:**
```bash
export NVFUSER_DUMP=python_definition
```
Example Output:
```python
def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, 1, -1], contiguity=[True, None, True], dtype=DataType.Float)
    T1 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[False, False, False], dtype=DataType.Float)
    S2 = fd.define_scalar(3.00000)
    T3 = fd.ops.add(T0, T1)
    T4 = fd.ops.mul(T3, S2)
    T5 = fd.ops.sum(T4, axes=[-1], keepdim=False, dtype=DataType.Float)
    fd.add_output(T5)
```
