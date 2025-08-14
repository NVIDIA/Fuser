import torch
from nvfuser import FusionDefinition


def test_slice_zero():
    with FusionDefinition() as fd:
        inp = fd.define_tensor(shape=[1, 16, 16, 1], contiguity=True)
        left = fd.ops.slice(
            inp,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 16, 16, 0],
            strides=[1, 1, 1, 1],
        )
        right = fd.ops.slice(
            inp,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 16, 16, 1],
            strides=[1, 1, 1, 1],
        )
        out = fd.ops.cat([left, right], dim=-1)
        fd.add_output(out)

    inp = torch.testing.make_tensor((1, 16, 16, 1), dtype=torch.float32, device="cuda")
    (out,) = fd.execute([inp])
    assert out.size() == (1, 16, 16, 1)
