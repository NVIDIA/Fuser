#!/usr/bin/env python3
"""
CutlassExecutor Example

This example demonstrates how to use the CutlassExecutor for nvfp4 scaled matrix multiplication.
The CutlassExecutor automatically generates and compiles CUTLASS kernels using nvcc,
then loads and executes them dynamically.

Requirements:
- NVIDIA GPU with compute capability 10.0+ (SM100+)
- CUDA 13.0+
- PyTorch with CUDA support
- nvFuser with CUTLASS support
"""

import torch
import nvfuser
import time

def create_nvfp4_scaled_matmul_fusion():
    """Create a fusion for nvfp4 scaled matrix multiplication."""
    with nvfuser.FusionDefinition() as fd:
        # Create input tensors
        # Note: In practice, these would be nvfp4 tensors, but for this example
        # we'll use BFloat16 to demonstrate the fusion structure
        tv0 = fd.define_tensor([-1, -1], contiguity=[True, True], dtype=nvfuser.DataType.BFloat16)  # Matrix A (M x K)
        tv1 = fd.define_tensor([-1, -1], contiguity=[True, True], dtype=nvfuser.DataType.BFloat16)  # Matrix B (N x K) 
        tv2 = fd.define_tensor([-1, -1], contiguity=[True, True], dtype=nvfuser.DataType.Float)  # Scale A
        tv3 = fd.define_tensor([-1, -1], contiguity=[True, True], dtype=nvfuser.DataType.Float)  # Scale B
        
        # Transpose B for column major layout
        tv1_t = fd.ops.permute(tv1, [1, 0])
        
        # Create scaled matmul operation
        scaled_out, _, _ = fd.ops.scaled_mm(tv0, tv1_t, tv2, tv3)
        
        fd.add_output(scaled_out)
    
    return fd

def run_cutlass_executor_example():
    """Run the CutlassExecutor example."""
    print("=== CutlassExecutor Example ===")
    print("This example demonstrates JIT compilation of CUTLASS kernels")
    print()
    
    # Check GPU capability
    if torch.cuda.get_device_capability() < (10, 0):
        print("❌ This example requires compute capability 10.0+ (SM100+)")
        print(f"Current GPU capability: {torch.cuda.get_device_capability()}")
        return
    
    print(f"✅ GPU capability: {torch.cuda.get_device_capability()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print()
    
    # Create fusion
    print("Creating nvfp4 scaled matmul fusion...")
    fusion_def = create_nvfp4_scaled_matmul_fusion()
    print("✅ Fusion created successfully")
    print()
    
    # Create test data
    print("Creating test data...")
    M, N, K = 512, 512, 256
    block_size = 16
    
    # Create input tensors
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    scale_a = torch.randn(M, 1, dtype=torch.float32, device='cuda')
    scale_b = torch.randn(1, N, dtype=torch.float32, device='cuda')
    
    inputs = [a, b, scale_a, scale_b]
    
    print(f"✅ Test data created: {M}x{K} @ {N}x{K}")
    print()
    
    # Execute the fusion
    print("Executing fusion with nvFuser...")
    start_time = time.time()
    
    try:
        outputs = fusion_def.execute(inputs)
        run_time = time.time() - start_time
        
        print(f"✅ Fusion execution successful in {run_time:.3f}s")
        print(f"✅ Output shape: {outputs[0].shape}")
        print()
        
        # Show performance metrics
        total_elements = M * N
        gflops = (2 * M * N * K) / (run_time * 1e9)
        print(f"Performance metrics:")
        print(f"  - Matrix size: {M}x{N}x{K}")
        print(f"  - Total elements: {total_elements:,}")
        print(f"  - Execution time: {run_time:.3f}s")
        print(f"  - Theoretical GFLOPS: {gflops:.1f}")
        print()
        
        # Show the generated CUDA code
        print("Generated CUDA code:")
        print("=" * 50)
        cuda_code = fusion_def.last_cuda_code()
        print(cuda_code[:1000] + "..." if len(cuda_code) > 1000 else cuda_code)
        print("=" * 50)
        print()
        
        print("🎉 CutlassExecutor example completed successfully!")
        print()
        print("Note: This example demonstrates the nvFuser fusion execution.")
        print("The CutlassExecutor is used internally by nvFuser when it detects")
        print("supported CUTLASS patterns like scaled matrix multiplication.")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print()
        print("This might be because:")
        print("1. CUTLASS is not properly installed")
        print("2. The fusion pattern is not supported")
        print("3. There's an issue with the JIT compilation pipeline")
        import traceback
        traceback.print_exc()
        return

def main():
    """Main function."""
    try:
        run_cutlass_executor_example()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
