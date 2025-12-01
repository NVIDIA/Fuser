#!/usr/bin/env python3
"""
Standalone test to reproduce at::linear hang with meta tensors.
This test is independent of nvfuser and only uses PyTorch.

Based on the debug output showing:
- Input: shape [2, 4, 16], strides [64, 16, 1], dtype bfloat16, device meta
- Weight: shape [16, 16], strides [16, 1], dtype bfloat16, device meta
"""

import torch
import sys


def print_tensor_info(name, tensor):
    """Print detailed tensor metadata."""
    print(f"\n[INFO] {name}:")
    print(f"  shape: {list(tensor.shape)}")
    print(f"  strides: {list(tensor.stride())}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  is_contiguous: {tensor.is_contiguous()}")
    print(f"  numel: {tensor.numel()}")
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("Testing at::linear with meta tensors (bfloat16)")
    print("=" * 80)
    sys.stdout.flush()
    
    # Create input tensor: [2, 4, 16] with strides [64, 16, 1], bfloat16, meta device
    print("\n[STEP 1] Creating input tensor...")
    sys.stdout.flush()
    
    input_tensor = torch.empty_strided(
        size=(2, 4, 16),
        stride=(64, 16, 1),
        dtype=torch.bfloat16,
        device='meta'
    )
    print_tensor_info("input_tensor", input_tensor)
    
    # Create weight tensor: [16, 16] with strides [16, 1], bfloat16, meta device
    print("\n[STEP 2] Creating weight tensor...")
    sys.stdout.flush()
    
    weight_tensor = torch.empty_strided(
        size=(16, 16),
        stride=(16, 1),
        dtype=torch.bfloat16,
        device='meta'
    )
    print_tensor_info("weight_tensor", weight_tensor)
    
    # Call torch.nn.functional.linear - this may hang
    print("\n[STEP 3] Calling torch.nn.functional.linear(input, weight)")
    print("         THIS MAY HANG!")
    print("-" * 80)
    sys.stdout.flush()
    
    try:
        result = torch.nn.functional.linear(input_tensor, weight_tensor)
        
        print("\n[SUCCESS] torch.nn.functional.linear completed!")
        sys.stdout.flush()
        
        print_tensor_info("result", result)
        
        # Verify expected shape
        expected_shape = (2, 4, 16)
        actual_shape = tuple(result.shape)
        print(f"\n[VERIFICATION] Expected shape: {expected_shape}")
        print(f"[VERIFICATION] Actual shape: {actual_shape}")
        
        if actual_shape == expected_shape:
            print("[VERIFICATION] ✓ Shape matches!")
        else:
            print("[VERIFICATION] ✗ Shape mismatch!")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        sys.stdout.flush()
        
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {type(e).__name__}: {e}")
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print()
    sys.stdout.flush()
    
    main()

