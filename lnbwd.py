
import torch
import os
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
PROMOTE_DTYPES = [DataType.BFloat16, DataType.Half]

def layernorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )

    T2 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T3 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )

    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)

    V8 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T9 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    V12 = T0.shape()
    T13 = fd.ops.broadcast_in_dim(T9, shape=V12, broadcast_dims=[0, 1])
    T14 = fd.ops.sub(T0, T13)

    T18 = fd.ops.broadcast_in_dim(T3, shape=V12, broadcast_dims=[0, 1])
    T19 = fd.ops.mul(T14, T18)

    T23 = fd.ops.broadcast_in_dim(T4, shape=V12, broadcast_dims=[1])
    T28 = fd.ops.sum(T1, dims=[0], keepdim=False, dtype=DataType.Null)

    T30 = fd.ops.mul(T1, T23)
    T31 = fd.ops.mul(T1, T19)
    T32 = fd.ops.sum(T31, dims=[0], keepdim=False, dtype=DataType.Null)

    T34 = fd.ops.mul(T30, T18)
    T35 = fd.ops.mul(T30, T14)
    T36 = fd.ops.sum(T35, dims=[1], keepdim=False, dtype=DataType.Null)

    T40 = fd.ops.broadcast_in_dim(T36, shape=V8, broadcast_dims=[0])
    T41 = fd.ops.neg(T34)
    T42 = fd.ops.sum(T41, dims=[1], keepdim=False, dtype=DataType.Null)
    T46 = fd.ops.broadcast_in_dim(T42, shape=V8, broadcast_dims=[0])
    S47 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T48 = fd.ops.mul(S47, T40)
    S49 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T50 = fd.ops.pow(T3, S49)
    T51 = fd.ops.mul(T48, T50)
    T54 = fd.ops.sum(T46, dims=[1], keepdim=False, dtype=DataType.Null)
    T55 = fd.ops.sum(T51, dims=[1], keepdim=False, dtype=DataType.Null)

    T59 = fd.ops.broadcast_in_dim(T55, shape=V8, broadcast_dims=[0])
    T63 = fd.ops.broadcast_in_dim(T59, shape=V12, broadcast_dims=[0, 1])
    T67 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    T71 = fd.ops.broadcast_in_dim(T67, shape=V12, broadcast_dims=[0, 1])

    S72 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T73 = fd.ops.mul(S72, T63)
    T74 = fd.ops.sub(T0, T71)
    T75 = fd.ops.mul(T73, T74)

    S77 = fd.ops.reciprocal(T0.size(1))
    T78 = fd.ops.mul(T75, S77)
    T82 = fd.ops.broadcast_in_dim(T54, shape=V8, broadcast_dims=[0])
    T86 = fd.ops.broadcast_in_dim(T82, shape=V12, broadcast_dims=[0, 1])
    T88 = fd.ops.mul(S77, T86)
    T89 = fd.ops.add(T78, T88)
    T90 = fd.ops.add(T34, T89)

    if dtype in PROMOTE_DTYPES:
        T28 = fd.ops.cast(T28, dtype=dtype)
        T90 = fd.ops.cast(T90, dtype=dtype)
        T32 = fd.ops.cast(T32, dtype=dtype)

    fd.add_output(T90)
    fd.add_output(T32)
    fd.add_output(T28)


if __name__ == "__main__":
    eps = 1e-5
    dim0 = 16384
    
    import argparse
    parser = argparse.ArgumentParser(description="Run the reduction model with a specified size.")
    parser.add_argument("dim1", type=int, help="The size parameter for the model")
    args = parser.parse_args()
    dim1 = args.dim1

    size = (dim0, dim1)
    dtype = torch.bfloat16
    batch_size, hidden_size = size
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    mean = inputs.to(torch.float).mean(dim=-1)
    variance = inputs.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

    with FusionDefinition() as fd:
        layernorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    is_validation = False
    if is_validation:
        eager_output = torch.nn.functional.layer_norm(
            inputs.to(torch.double),
            inputs.shape[1:],
            weight=weights.to(torch.double),
            bias=bias.to(torch.double),
        )
        eager_output.backward(grads.to(torch.double))
        fd.validate(
            [inputs, grads, mean, invstd, weights],
            [inputs.grad, weights.grad, bias.grad],
        )
    else:
        out = fd.execute([inputs, grads, mean, invstd, weights])

    is_profiling = False
    if is_profiling:
      out = fd.execute([inputs, grads, mean, invstd, weights], profile=is_profiling)
      prof = fd.profile()
      peak_percentage = prof.kernel_profiles[0].percentage_peak_bandwidth
      registers = prof.kernel_profiles[0].registers
      time_ms = prof.kernel_profiles[0].time_ms
      grid_str = prof.kernel_profiles[0].grid_str
      block_str = prof.kernel_profiles[0].block_str
      print(f"time_ms {time_ms:.3f} Peak_bandwidth_percentage {peak_percentage:.1f} Registers {registers} grid {grid_str} block {block_str}")

# 16K * 2048
# Fus#  NSegs CuEvtTm(ms) HstTm(ms) CmpTm(ms) KerTm(ms) EffBw(GB/s) %PkBw   S-Seg# S-KerTm(ms) S-EffBw(GB/s) S-%PkBw S-In(MB) S-Out(MB) S-Smem[Dyn,Stat] S-Regs S-Grid           S-Block          S-KerName           
#     0     1     886.349   886.311   822.817     0.059    3395.866   41.45      0       0.058      3465.017   42.30  134.353    67.117        [512, 16]     96      [1, 740, 1]       [2, 64, 1] nvfuser_inner_outer_persistent_f0_c1_r0_g0

# SMEMBUFFER=2 WAVES=8 VECT=8
# Fus#  NSegs CuEvtTm(ms) HstTm(ms) CmpTm(ms) KerTm(ms) EffBw(GB/s) %PkBw   S-Seg# S-KerTm(ms) S-EffBw(GB/s) S-%PkBw S-In(MB) S-Out(MB) S-Smem[Dyn,Stat] S-Regs S-Grid           S-Block          S-KerName           
#     0     1     850.220   850.165   787.114     0.070    2885.397   35.22      0       0.069      2936.537   35.85  134.353    67.117       [8704, 16]     64     [1, 1184, 1]      [1, 128, 1] nvfuser_inner_outer_persistent_f0_c1_r0_g0

# VECT=16
# Fus#  NSegs CuEvtTm(ms) HstTm(ms) CmpTm(ms) KerTm(ms) EffBw(GB/s) %PkBw   S-Seg# S-KerTm(ms) S-EffBw(GB/s) S-%PkBw S-In(MB) S-Out(MB) S-Smem[Dyn,Stat] S-Regs S-Grid           S-Block          S-KerName           
#     0     1     771.478   771.435   708.643     0.059    3414.282   41.68      0       0.058      3486.122   42.56  134.353    67.117        [512, 16]     96      [1, 740, 1]       [2, 64, 1] nvfuser_inner_outer_persistent_f0_c1_r0_g0

# inner reduction + partial outer reduction
# Fus#  NSegs CuEvtTm(ms) HstTm(ms) CmpTm(ms) KerTm(ms) EffBw(GB/s) %PkBw   S-Seg# S-KerTm(ms) S-EffBw(GB/s) S-%PkBw S-In(MB) S-Out(MB) S-Smem[Dyn,Stat] S-Regs S-Grid           S-Block          S-KerName           
#     0     1     824.548   824.512   761.035     0.054    3743.125   45.69      0       0.053      3829.645   46.75  134.353    67.117        [512, 16]     96      [1, 740, 1]       [2, 64, 1] nvfuser_inner_outer_persistent_f0_c1_r0_g0

# inner reduction only
# Fus#  NSegs CuEvtTm(ms) HstTm(ms) CmpTm(ms) KerTm(ms) EffBw(GB/s) %PkBw   S-Seg# S-KerTm(ms) S-EffBw(GB/s) S-%PkBw S-In(MB) S-Out(MB) S-Smem[Dyn,Stat] S-Regs S-Grid           S-Block          S-KerName           
#     0     1     767.234   767.205   703.670     0.046    4405.833   53.78      0       0.045      4526.194   55.25  134.353    67.117        [512, 16]     72      [1, 740, 1]       [2, 64, 1] nvfuser_inner_outer_persistent_f0_c1_r0_g0


# inner reduction only + disable magic zero
# Fus#  NSegs CuEvtTm(ms) HstTm(ms) CmpTm(ms) KerTm(ms) EffBw(GB/s) %PkBw   S-Seg# S-KerTm(ms) S-EffBw(GB/s) S-%PkBw S-In(MB) S-Out(MB) S-Smem[Dyn,Stat] S-Regs S-Grid           S-Block          S-KerName           
#     0     1     756.755   756.724   692.993     0.046    4375.216   53.41      0       0.045      4493.887   54.86  134.353    67.117        [512, 16]     72      [1, 740, 1]       [2, 64, 1] nvfuser_inner_outer_persistent_f0_c1_r0_g0