**nvFuser**: A Deep Learning Compiler for PyTorch on NVIDIA GPUs

**nvFuser** is a specialized compiler developed by NVIDIA to accelerate PyTorch deep learning models running on NVIDIA GPUs (Volta architecture and newer). It achieves this by automatically generating highly optimized, custom "fusion" kernels using just-in-time (JIT) compilation.

**Key Concepts:**

1.  **Compiler:** `nvFuser` translates sequences of PyTorch operations into efficient low-level GPU code.
2.  **Operation Fusion:** It intelligently combines multiple compatible PyTorch operations (e.g., element-wise math, reductions, normalizations) into a single, unified GPU kernel. This fusion process minimizes the overhead associated with launching multiple kernels and reduces data transfers between the GPU's global memory and its processing units. The result is often significant performance gains, particularly for operations limited by memory bandwidth.
3.  **Just-in-Time (JIT) Compilation:** The optimized kernels are compiled during the program's execution ("just-in-time"). This allows `nvFuser` to create kernels specifically tailored to the actual input shapes, data types, and hardware characteristics encountered at runtime, providing flexibility for models with dynamic inputs.
4.  **PyTorch Integration:** Modern high-performance PyTorch frameworks utilize `nvFuser` as a backend optimization engine. A prominent example is **`Lightning Thunder`**. `Thunder` acts as a source-to-source compiler for PyTorch, analyzing Python code, capturing the computational graph, and dispatching segments of this graph to specialized backends like `nvFuser` for optimized execution on NVIDIA GPUs.

**In Summary:**

`nvFuser` is the core NVIDIA technology providing JIT fusion capabilities to accelerate PyTorch workloads on GPUs. High-level frameworks like `Lightning Thunder` leverage `nvFuser` (and other tools) to automatically optimize PyTorch programs for maximum performance.

**Source Code:** [https://github.com/NVIDIA/Fuser](https://github.com/NVIDIA/Fuser)

## C++ API Example

Below is a simplified C++ example demonstrating how to define and schedule a simple element-wise operation using the nvFuser API. This example is adapted from the test suite.

```cpp
#include <fusion.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>

// ... other necessary includes ...

void simple_pointwise_example() {
    // 1. Create a Fusion object to hold the computation graph.
    nvfuser::Fusion fusion;
    // 2. Use FusionGuard to set this fusion as the active one for IR building.
    nvfuser::FusionGuard fg(&fusion);

    // 3. Define Input Tensors:
    //    Create symbolic tensor views representing the inputs.
    //    'makeContigTensor(nDims)' creates a contiguous tensor of nDims with default float32 type.
    int nDims = 2;
    nvfuser::TensorView* tv0 = nvfuser::makeContigTensor(nDims);
    nvfuser::TensorView* tv1 = nvfuser::makeContigTensor(nDims);

    // 4. Register Inputs:
    //    Mark the created tensors as inputs to the fusion.
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    // 5. Define Computation:
    //    Perform element-wise operations. 'add' creates an addition node.
    //    'IrBuilder::create<Val>(2.0)' creates a scalar constant.
    nvfuser::TensorView* tv2 = nvfuser::add(tv1, nvfuser::IrBuilder::create<nvfuser::Val>(2.0));
    nvfuser::TensorView* tv3 = nvfuser::add(tv0, tv2);

    // 6. Register Output:
    //    Mark the final tensor as an output of the fusion.
    fusion.addOutput(tv3);

    // 7. Scheduling Transformations:
    //    Apply transformations to optimize the execution on the GPU.
    //    Transformations operate on the domains of the TensorViews.

    //    Merge the two dimensions of the output tensor tv3 into one.
    //    Original: [I0, I1] -> After merge(0): [I0*I1]
    tv3->merge(0);

    //    Split the merged dimension. Let's say we want blocks of size 128.
    //    [I0*I1] -> split(0, 128): [ceilDiv(I0*I1, 128), 128]
    tv3->split(0, 128);

    //    Optionally, apply further splits, e.g., for unrolling.
    //    [ceilDiv(I0*I1, 128), 128] -> split(0, 4): [ceilDiv(I0*I1, 128*4), 4, 128]
    tv3->split(0, 4);

    // 8. Apply ComputeAt and Inlining (Common Scheduling Steps):
    //    Define where intermediate tensors are computed relative to the output.
    //    This helps control loop nesting and fusion.
    //    'ComputeAtMode::MostInlined' attempts to compute producers as late as possible.
    nvfuser::TensorView* tv_inputs[] = {tv0, tv1};
    for(auto tv_input : tv_inputs) {
        tv_input->computeAt(tv3, -1, nvfuser::ComputeAtMode::MostInlined);
    }

    //    Inline element-wise operations where possible.
    nvfuser::inlineMost();

    // 9. Parallelization:
    //    Map tensor axes to GPU hardware dimensions (BlockIdx, ThreadIdx).
    tv3->axis(0)->parallelize(nvfuser::ParallelType::BIDx); // Outer dimension to BlockIdx.x
    tv3->axis(1)->parallelize(nvfuser::ParallelType::Unroll); // Middle dimension for loop unrolling
    tv3->axis(2)->parallelize(nvfuser::ParallelType::TIDx);  // Inner dimension to ThreadIdx.x

    //    Propagate parallelization to producers (often needed after computeAt).
    nvfuser::scheduler_utils::parallelizeAllLike(tv3);

    // 10. Compilation and Execution (Conceptual):
    //    The fusion definition is now ready. The next steps (not shown here)
    //    would involve:
    //    - Lowering the Fusion IR to Kernel IR (GpuLower).
    //    - Generating CUDA code (codegen::generateCudaKernel).
    //    - Compiling the CUDA code (nvrtc).
    //    - Executing the compiled kernel with actual input tensors (KernelExecutor).
}

### How Scheduling Optimizes the Code

The scheduling steps (`merge`, `split`, `computeAt`, `inlineMost`, `parallelize`) are crucial for performance. They transform the initial, straightforward representation of the computation into a structure optimized for GPU execution. Here's a conceptual breakdown:

*   **Without Scheduling (Conceptual):** nvFuser might generate separate, simple kernels for each operation:

    ```cuda
    // Kernel 1: Compute tv2 = tv1 + 2.0
    __global__ void kernel1(float* tv1, float* tv2, /*...sizes...*/) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            tv2[idx] = tv1[idx] + 2.0f;
        }
    }

    // Kernel 2: Compute tv3 = tv0 + tv2
    __global__ void kernel2(float* tv0, float* tv2, float* tv3, /*...sizes...*/) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            tv3[idx] = tv0[idx] + tv2[idx];
        }
    }

    // Host code would launch kernel1, wait, then launch kernel2.
    // tv2 is written to and read from global memory.
    ```

*   **With Scheduling (Conceptual):** The scheduling steps aim to create a single, optimized, fused kernel:

    ```cuda
    __global__ void fused_kernel(float* tv0, float* tv1, float* tv3, /*...sizes...*/) {
        // blockIdx.x determined by tv3->axis(0)->parallelize(BIDx)
        // threadIdx.x determined by tv3->axis(2)->parallelize(TIDx)
        // The loop for the unrolled axis (tv3->axis(1)) is expanded by the compiler.

        for (int unroll_idx = 0; unroll_idx < 4; ++unroll_idx) { // Loop from tv3->axis(1) (Unroll)
            // Calculate global index based on blockIdx, threadIdx, and unroll_idx
            int idx = calculate_global_index(blockIdx.x, threadIdx.x, unroll_idx, /*...sizes...*/);

            if (idx < total_elements) {
                // Inlining/Fusion via computeAt:
                // tv2 = tv1 + 2.0 is computed directly here, potentially using registers.
                float tv2_val = tv1[idx] + 2.0f;
                // tv3 = tv0 + tv2 is computed using the register value.
                tv3[idx] = tv0[idx] + tv2_val;
            }
        }
    }

    // Host code launches just one kernel.
    // Intermediate tv2 potentially lives only in registers, avoiding global memory.
    ```

**Summary of Impacts:**

1.  **`merge`/`split`:** Directly define the **loop structure** (nesting, bounds) in the CUDA kernel.
2.  **`computeAt`/`inlineMost`:** Enable **operator fusion**, putting multiple operations within the same loop nest. This reduces kernel launches and keeps intermediate data in **registers/shared memory**, minimizing slow global memory access.
3.  **`parallelize`:** Maps the abstract loops defined by merge/split onto the physical **GPU threads and blocks** (`threadIdx`, `blockIdx`) and utilizes hardware features like **vectorization** (`float4`) or **loop unrolling**. 