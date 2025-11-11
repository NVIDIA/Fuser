import torch

from nvfuser_direct import FusionDefinition, DataType


def test_pointwise():
    with FusionDefinition() as fd:
        x = fd.define_tensor([-1, -1], dtype=DataType.Float)
        x = fd.ops.relu(x)
        fd.add_output(x)

    fd.fusion.print()

    x = torch.randn(2, 3, device="cuda")
    [y] = fd.execute([x])
    torch.testing.assert_close(y, x.relu())


def test_symbolic_reshape():
    with FusionDefinition() as fd:
        x = fd.define_tensor([-1, -1], dtype=DataType.Float)
        x = fd.ops.reshape(x, [-1])
        x = fd.ops.reshape(x, [-1, 2])
        fd.add_output(x)

    fd.fusion.print()

    x = torch.arange(6, dtype=torch.float32, device="cuda").view(2, 3)
    [y] = fd.execute([x])
    torch.testing.assert_close(y, x.view(-1, 2))


def test_concrete_reshape():
    with FusionDefinition() as fd:
        x = fd.define_tensor([2, 3], dtype=DataType.Float)
        x = fd.ops.reshape(x, [-1])
        x = fd.ops.reshape(x, [-1, 2])
        fd.add_output(x)

    fd.fusion.print()

    x = torch.arange(6, dtype=torch.float32, device="cuda").view(2, 3)
    [y] = fd.execute([x])
    torch.testing.assert_close(y, x.view(-1, 2))
