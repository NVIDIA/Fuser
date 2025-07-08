# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Callable
import torch_ops as ops

# Global flag to control debug output
DEBUG_IOBYTES = False

def set_debug_iobytes(enabled: bool):
  """Enable or disable debug output for iobytes calculations"""
  global DEBUG_IOBYTES
  DEBUG_IOBYTES = enabled

def total_bytes(obj) -> int:
  if isinstance(obj, torch.Tensor):
    return obj.element_size() * obj.nelement()
  elif isinstance(obj, (list, tuple)):
    return sum(total_bytes(t) for t in obj if isinstance(t, torch.Tensor))
  elif isinstance(obj, dict):
    return sum(total_bytes(t) for t in obj.values() if isinstance(t, torch.Tensor))
  else:
    return 0

def get_iobytes_for_fwd(model: Callable, inputs: list) -> int:
  if DEBUG_IOBYTES:
    print("\n=== get_iobytes_for_fwd DEBUG ===")
    
  output = model(inputs)
  
  # Calculate input bytes
  input_bytes = total_bytes(inputs)
  if DEBUG_IOBYTES:
    print(f"Inputs:")
    for i, inp in enumerate(inputs):
      if isinstance(inp, torch.Tensor):
        print(f"  Input {i}: shape={inp.shape}, dtype={inp.dtype}, bytes={total_bytes(inp)}")
      else:
        print(f"  Input {i}: {type(inp).__name__} = {inp}")
    print(f"  Total input bytes: {input_bytes}")
  
  # Calculate output bytes
  output_bytes = total_bytes(output)
  if DEBUG_IOBYTES:
    print(f"\nOutput:")
    if isinstance(output, torch.Tensor):
      print(f"  shape={output.shape}, dtype={output.dtype}, bytes={output_bytes}")
    elif isinstance(output, (list, tuple)):
      print(f"  {len(output)} outputs, total bytes={output_bytes}")
    else:
      print(f"  type={type(output).__name__}, bytes={output_bytes}")

  iobytes = input_bytes + output_bytes
  grad_fn = None
  if isinstance(output, torch.Tensor) and hasattr(output, 'grad_fn'):
    grad_fn = output.grad_fn
  
  if grad_fn is None:
    if DEBUG_IOBYTES:
      print(f"\nNo grad_fn found (requires_grad=False)")
      print(f"Total iobytes: {iobytes}")
    return iobytes
  
  if DEBUG_IOBYTES:
    print(f"\ngrad_fn: {grad_fn}")
    print(f"Analyzing saved tensors:")
  
  saved_count = 0
  saved_bytes = 0
  for attr in grad_fn.__dir__():
    if not attr.startswith("_saved"):
      continue
    saved_var = getattr(grad_fn, attr)
    if not isinstance(saved_var, torch.Tensor):
      continue
    
    # Check if it's the same object as one of the inputs
    is_input = False
    input_idx = -1
    for idx, inp in enumerate(inputs if isinstance(inputs, (list, tuple)) else [inputs]):
      if isinstance(inp, torch.Tensor) and saved_var is inp:
        is_input = True
        input_idx = idx
        break
    
    if is_input:
      if DEBUG_IOBYTES:
        print(f"  {attr}: shape={saved_var.shape}, bytes={total_bytes(saved_var)} -> SKIP (same as input {input_idx})")
      continue
    
    saved_count += 1
    bytes_count = total_bytes(saved_var)
    saved_bytes += bytes_count
    
    if DEBUG_IOBYTES:
      print(f"  {attr}: shape={saved_var.shape}, dtype={saved_var.dtype}, bytes={bytes_count} -> COUNTED")
    
    iobytes += bytes_count
  
  if DEBUG_IOBYTES:
    print(f"\nSaved tensors summary:")
    print(f"  Total saved: {saved_count}")
    print(f"  Total saved bytes: {saved_bytes}")
    print(f"\nFinal iobytes breakdown:")
    print(f"  Inputs: {input_bytes}")
    print(f"  Output: {output_bytes}")
    print(f"  Saved: {saved_bytes}")
    print(f"  Total: {iobytes}")
  
  return iobytes

def get_iobytes_for_bwd(output: torch.Tensor) -> int:
  if DEBUG_IOBYTES:
    print("\n=== get_iobytes_for_bwd DEBUG ===")
    
  grad_fn = output.grad_fn
  assert grad_fn is not None, "Output must have grad_fn for backward pass"
  
  # 2 * output because we need to store the output and the grad_output
  output_bytes = total_bytes(output)
  iobytes = 2 * output_bytes
  
  if DEBUG_IOBYTES:
    print(f"Output: shape={output.shape}, dtype={output.dtype}")
    print(f"  Output bytes: {output_bytes}")
    print(f"  With grad_output: {2 * output_bytes}")
    print(f"\ngrad_fn: {grad_fn}")
    print(f"Analyzing saved tensors:")
  
  saved_count = 0
  saved_with_grad = 0
  saved_bytes_total = 0
  
  for attr in grad_fn.__dir__():
    if not attr.startswith("_saved"):
      continue
    saved_var = getattr(grad_fn, attr)
    if isinstance(saved_var, torch.Tensor):
      requires_grad = saved_var.requires_grad
      bytes_count = total_bytes(saved_var)
      multiplier = requires_grad + 1
      bytes_with_grad = bytes_count * multiplier
      
      saved_count += 1
      saved_bytes_total += bytes_with_grad
      if requires_grad:
        saved_with_grad += 1
      
      if DEBUG_IOBYTES:
        print(f"  {attr}: shape={saved_var.shape}, dtype={saved_var.dtype}, requires_grad={requires_grad}")
        print(f"    bytes={bytes_count}, multiplier={multiplier}, total={bytes_with_grad}")
      
      iobytes += bytes_with_grad
  
  if DEBUG_IOBYTES:
    print(f"\nSaved tensors summary:")
    print(f"  Total saved: {saved_count}")
    print(f"  With requires_grad: {saved_with_grad}")
    print(f"  Total saved bytes (with multipliers): {saved_bytes_total}")
    print(f"\nFinal iobytes breakdown:")
    print(f"  Output + grad_output: {2 * output_bytes}")
    print(f"  Saved tensors: {saved_bytes_total}")
    print(f"  Total: {iobytes}")
  
  return iobytes

if __name__ == "__main__":
  # Enable debug for testing
  set_debug_iobytes(True)
  
  inputs = torch.randn(8, 16, requires_grad=True)
  weights = torch.ones(16, requires_grad=True)
  bias = torch.ones(16, requires_grad=True)
  
  def model(fwd_inputs):
    return ops.layernorm(fwd_inputs)
  fwd_inputs = [inputs, weights, bias]
  print("Testing forward pass:")
  print(get_iobytes_for_fwd(model, fwd_inputs))
  
  print("\n\nTesting backward pass:")
  output = model(fwd_inputs)
  print(get_iobytes_for_bwd(output))

  

