import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id23(fd : FusionDefinition) -> None :
    S0 = fd.define_scalar(None, dtype=DataType.Double)
    S1 = fd.define_scalar(None, dtype=DataType.Double)
    S2 = fd.define_scalar(None, dtype=DataType.Double)
    T3 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    T4 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False)
    T5 = fd.ops.cast(T4, dtype=DataType.Float)
    T6 = fd.ops.mul(T5, T5)
    T7 = fd.ops.mul(T6, T5)
    S8 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T9 = fd.ops.mul(S8, T5)
    S10 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T11 = fd.ops.mul(S10, T7)
    T12 = fd.ops.add(T5, T11)
    S13 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T14 = fd.ops.mul(S13, T12)
    T15 = fd.ops.tanh(T14)
    S16 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T17 = fd.ops.add(S16, T15)
    S18 = fd.define_scalar(16, dtype=DataType.Int)
    S19 = fd.define_scalar(128, dtype=DataType.Int)
    S20 = fd.define_scalar(3072, dtype=DataType.Int)
    V21 = fd.define_vector([S18, S19, S20], dtype=DataType.Int)
    T22 = fd.ops.reshape(T3, new_shape=V21)
    T23 = fd.ops.cast(T22, dtype=DataType.Float)
    T24 = fd.ops.mul(T23, T17)
    T25 = fd.ops.mul(T23, T9)
    T26 = fd.ops.mul(T15, T15)
    S27 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T28 = fd.ops.sub(S27, T26)
    T29 = fd.ops.mul(T25, T28)
    T30 = fd.ops.mul(T29, S2)
    T31 = fd.ops.mul(T30, S1)
    T32 = fd.ops.mul(T24, S0)
    T33 = fd.ops.add(T30, T32)
    T34 = fd.ops.mul(T31, T5)
    T35 = fd.ops.mul(T31, T6)
    T36 = fd.ops.add(T33, T35)
    T37 = fd.ops.mul(T34, T5)
    T38 = fd.ops.add(T36, T37)
    T39 = fd.ops.add(T38, T37)
    T40 = fd.ops.cast(T39, dtype=DataType.BFloat16)
    S41 = fd.define_scalar(2048, dtype=DataType.Int)
    S42 = fd.define_scalar(3072, dtype=DataType.Int)
    V43 = fd.define_vector([S41, S42], dtype=DataType.Int)
    T44 = fd.ops.reshape(T40, new_shape=V43)
    S45 = fd.define_scalar(2048, dtype=DataType.Int)
    S46 = fd.define_scalar(3072, dtype=DataType.Int)
    V47 = fd.define_vector([S45, S46], dtype=DataType.Int)
    T48 = fd.ops.reshape(T40, new_shape=V47)
    T49 = fd.ops.permute(T44, dims=[1, 0])
    T50 = fd.ops.sum(T39, axes=[0, 1], keepdim=False, dtype=DataType.Null)
    T51 = fd.ops.cast(T50, dtype=DataType.BFloat16)
    fd.add_output(T44)
    fd.add_output(T49)
    fd.add_output(T51)


if __name__ == "__main__":
    with FusionDefinition() as fd:
        nvfuser_fusion_id23(fd)

    out0, out1, _ = fd.execute([
        1.0,
        1.0,
        1.0,
        torch.randn([16 * 128, 3072], dtype=torch.bfloat16).cuda(),
        torch.randn([16, 128, 3072], dtype=torch.bfloat16).cuda()])
    print(out0.data_ptr())
    print(out1.data_ptr())
