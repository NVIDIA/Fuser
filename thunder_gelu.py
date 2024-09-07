import torch
import thunder


def fn(a):
    return torch.nn.functional.gelu(a, approximate="tanh")


batch_dim = 512
inner_dim = 1024
a = torch.randn(batch_dim, inner_dim, dtype=torch.bfloat16, device="cuda")

jfn = thunder.jit(fn)
o = jfn(a)
print(o)

extrace = thunder.last_traces(jfn)[-1]
print(extrace)
