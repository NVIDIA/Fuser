import nvfuser
from nvfuser import FusionDefinition, DataType
import torch
import torch.nn.functional as F
import thunder
from thunder.executors.nvfuserex import nvfuserex
from thunder.examine import get_fusions

def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[8192], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0]) # weight
    T2 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T3 = fd.ops.cast(T0, dtype=DataType.Float)
    T7 = fd.ops.broadcast_in_dim(T1, shape=[2048, 8192], broadcast_dims=[1])
    T8 = fd.ops.cast(T7, dtype=DataType.Float)
    T9 = fd.ops.mul(T8, T3)
    T10 = fd.ops.cast(T2, dtype=DataType.Float)
    T11 = fd.ops.mul(T10, T10)
    T12 = fd.ops.sum(T11, dims=[1], keepdim=False, dtype=DataType.Null)
    T16 = fd.ops.broadcast_in_dim(T12, shape=[2048, 1], broadcast_dims=[0])
    S17 = fd.define_scalar(8192.00, dtype=DataType.Double)
    S18 = fd.ops.reciprocal(S17)
    T19 = fd.ops.mul(T16, S18)
    S20 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T21 = fd.ops.add(T19, S20)
    T22 = fd.ops.rsqrt(T21)
    T23 = fd.ops.cast(T22, dtype=DataType.BFloat16)
    T27 = fd.ops.broadcast_in_dim(T23, shape=[2048, 8192], broadcast_dims=[0, 1])
    T28 = fd.ops.cast(T27, dtype=DataType.Float)
    T29 = fd.ops.mul(T10, T28)
    T30 = fd.ops.mul(T29, T3)
    T31 = fd.ops.sum(T30, dims=[0], keepdim=False, dtype=DataType.Null)
    T32 = fd.ops.cast(T31, dtype=DataType.BFloat16)
    T33 = fd.ops.mul(T28, T9)
    T34 = fd.ops.mul(T10, T9)
    T35 = fd.ops.sum(T34, dims=[1], keepdim=False, dtype=DataType.Null)
    T36 = fd.ops.cast(T35, dtype=DataType.BFloat16)
    T40 = fd.ops.broadcast_in_dim(T36, shape=[2048, 1], broadcast_dims=[0])
    T41 = fd.ops.cast(T40, dtype=DataType.Float)
    S42 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T43 = fd.ops.mul(S42, T41)
    S44 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T45 = fd.ops.pow(T22, S44)
    T46 = fd.ops.mul(T43, T45)
    S47 = fd.define_scalar(8192.00, dtype=DataType.Double)
    S48 = fd.ops.reciprocal(S47)
    T49 = fd.ops.mul(T46, S48)
    T50 = fd.ops.sum(T49, dims=[1], keepdim=False, dtype=DataType.Null)
    T54 = fd.ops.broadcast_in_dim(T50, shape=[2048, 1], broadcast_dims=[0])
    T58 = fd.ops.broadcast_in_dim(T54, shape=[2048, 8192], broadcast_dims=[0, 1])
    T59 = fd.ops.mul(T10, T58)
    T60 = fd.ops.add(T33, T59)
    T61 = fd.ops.add(T60, T59)
    T62 = fd.ops.cast(T61, dtype=DataType.BFloat16)
    fd.add_output(T32)
    fd.add_output(T62)

def rmsnorm_prims(inputs):
    inp, weights = inputs
    squared_mean = (inp**2).mean(1, keepdim=True)
    rms_eps = torch.sqrt(squared_mean + 1e-5)
    output = weights * (inp / rms_eps)
    return output

def rmsnorm_func(inputs):
    inp, weights = inputs
    output = F.rms_norm(
      inp,
      inp.shape[1:],
      weight=weights,
      eps = 1e-5
    )
    return output
  
def run_thunder_func():
  size = (2048, 8192)
  dtype =torch.bfloat16
  inputs = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
  grads = torch.randn(size, device="cuda", dtype=dtype)
  weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

  # Compile the fwd fn for torchcompile
  fwd_fn = thunder.jit(rmsnorm_func, executors=[nvfuserex])
  fwd_inputs = [inputs, weights]
  outputs = fwd_fn(fwd_inputs)
  outputs.backward(grads)
  
  print (thunder.__version__)
  print (nvfuser.__version__)
  fwd_trace = thunder.last_traces(fwd_fn)[-1]
  bwd_trace = thunder.last_backward_traces(fwd_fn)[-1]
  
  print (fwd_trace)
  print (bwd_trace)
  bwd_fusion = get_fusions(bwd_trace)
  print (bwd_fusion[-1][-1].last_used)

run_thunder_func()