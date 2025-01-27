import torch
from nvfuser import FusionDefinition, DataType


def a(fd: FusionDefinition):
    b = fd.define_tensor([1, 32, True])
    # Curiously these empty/useless define_tensors are critical to the bug!
    fd.define_tensor([1, 6, 2048], [None, True, True], DataType.BFloat16)
    d = fd.define_tensor([], [], DataType.BFloat16, stride_order=[0])
    fd.define_tensor([], [], DataType.BFloat16, stride_order=[1, 0])
    fd.define_tensor([], [], DataType.BFloat16, stride_order=[1, 0])
    e = fd.define_tensor([], [], DataType.BFloat16, stride_order=[1, 0])
    f = fd.ops.permute(b, [0, 2, 1])
    g = fd.ops.cat([f, f], -1)
    ab = fd.ops.cast(g, DataType.BFloat16)
    l = fd.ops.broadcast_in_dim(d, [1, 6, 2048], [2])
    n = fd.ops.mul(l, l)
    o = fd.ops.cast(n, DataType.BFloat16)
    p = fd.ops.linear(o, e)
    ah = fd.ops.reshape(p, [1, 6, 8, 64])
    ai = fd.ops.permute(ah, [0, 2, 1, 3])

    al = fd.ops.broadcast_in_dim(g, [1, 1, 6, 64], [0, 2, 3])
    am = fd.ops.broadcast_in_dim(ab, [1, 1, 6, 64], [0, 2, 3])
    bc = fd.ops.broadcast_in_dim(al, [1, 8, 6, 64], [0, 1, 2, 3])
    bf = fd.ops.mul(ai, bc)
    bg = fd.ops.slice(ai, [0, 0, 0, 0], [1, 8, 6, 32])
    bh = fd.ops.slice(ai, [0, 0, 0, 32], [1, 8, 6, 64])
    bl = fd.ops.cat([bh, bg], -1)
    bp = fd.ops.mul(bl, am)
    bq = fd.ops.add(bf, bp)
    br = fd.ops.cast(bq, DataType.BFloat16)
    bs = fd.ops.broadcast_in_dim(br, [1, 8, 1, 6, 64], [0, 1, 3, 4])
    bt = fd.ops.broadcast_in_dim(bs, [1, 8, 4, 6, 64], [0, 1, 2, 3, 4])
    bu = fd.ops.reshape(bt, [1, 32, 6, 64])
    by = fd.ops.stride_order(bu, [3, 2, 1, 0])
    fd.add_output(by)


with FusionDefinition() as fd:
    a(fd)

inputs = [
    torch.testing.make_tensor(1, 32, 6, dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor(1, 6, 2048, dtype=torch.bfloat16, device="cuda:0"),
    torch.testing.make_tensor(2048, dtype=torch.bfloat16, device="cuda:0"),
    torch.testing.make_tensor(2048, 2048, dtype=torch.bfloat16, device="cuda:0"),
    torch.testing.make_tensor(2, 2048, dtype=torch.bfloat16, device="cuda:0"),
    torch.testing.make_tensor(512, 2048, dtype=torch.bfloat16, device="cuda:0"),
]


fd.execute(inputs)