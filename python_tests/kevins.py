import torch
import nvfuser as nvf

def nvfuser_fusion_id4(fd : nvf.FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True],
                          dtype=nvf.DataType.BFloat16, is_cpu=False)
    T1 = fd.ops.cast(T0, dtype=nvf.DataType.Float)
    T2 = fd.ops.mul(T1, T1)
    T3 = fd.ops.mul(T2, T1)
    S4 = fd.define_scalar(0.500000, dtype=nvf.DataType.Double)
    T5 = fd.ops.mul(S4, T1)
    S6 = fd.define_scalar(0.0447150, dtype=nvf.DataType.Double)
    T7 = fd.ops.mul(S6, T3)
    T8 = fd.ops.add(T1, T7)
    S9 = fd.define_scalar(0.797885, dtype=nvf.DataType.Double)
    T10 = fd.ops.mul(S9, T8)
    T11 = fd.ops.tanh(T10)
    S12 = fd.define_scalar(1.00000, dtype=nvf.DataType.Double)
    T13 = fd.ops.add(S12, T11)
    T14 = fd.ops.mul(T5, T13)
    T15 = fd.ops.cast(T14, dtype=nvf.DataType.BFloat16)
    fd.add_output(T15)

torch.cuda.nvtx.range_push("Definition")
with nvf.FusionDefinition() as fd:
    nvfuser_fusion_id4(fd)
torch.cuda.nvtx.range_pop()

inputs = [[torch.randn(64, seq_len, 1600*4, device='cuda', dtype=torch.bfloat16)] for seq_len in range(2, 128, 2)]

print("Running for all", len(inputs), "inputs")
print(type(inputs[0][0]))
for idx in range(len(inputs)):
    s = "; ".join([str(t.shape) for t in inputs[idx]])
    s = "FusionDef::execute(" + s + ")"
    torch.cuda.nvtx.range_push(s)
    out = fd.execute(inputs[idx])
    torch.cuda.nvtx.range_pop()

print("Serializing...")
cache = nvf.FusionCache.get()
cache.serialize("fusion.cache")
print("Serialized.")
