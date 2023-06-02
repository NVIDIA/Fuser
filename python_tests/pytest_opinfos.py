import torch
from pytest_core import elementwise_unary_generator, _elementwise_unary_torch, OpInfo

opinfos = []

elementwise_unary_ops = []
acos_opinfo = OpInfo(
    lambda fd: fd.ops.acos,
    "acos",
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.acos),
)
elementwise_unary_ops.append(acos_opinfo)

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)
