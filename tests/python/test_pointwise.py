import pytest
import torch
from nvfuser import FusionDefinition, DataType


def test_issue_2395():
    def create_fusion(fd: FusionDefinition) -> None:
        cond0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, None],
            dtype=DataType.Bool,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        values = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        cond1 = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.Bool,
            is_cpu=False,
            stride_order=[1, 0],
        )
        cond1 = fd.ops.broadcast_in_dim(
            cond1, shape=[16, 16, 32], broadcast_dims=[0, 1]
        )
        sliced = fd.ops.slice(
            values,
            start_indices=[0, 0, 16],
            end_indices=[16, 16, 32],
            strides=[1, 1, 1],
        )
        zero = fd.define_scalar(0.00000, dtype=DataType.Double)
        masked = fd.ops.where(cond1, zero, values)
        masked = fd.ops.where(cond0, zero, masked)
        fd.add_output(sliced)
        fd.add_output(masked)

    with FusionDefinition() as fd:
        create_fusion(fd)
    ins = [
        torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda:0").as_strided(
            (16, 16, 32), (16, 1, 0)
        ),
        torch.randn((8192,), dtype=torch.float32, device="cuda:0").as_strided(
            (16, 16, 32), (512, 32, 1)
        ),
        torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda:0").as_strided(
            (16, 16), (16, 1)
        ),
    ]
    outs = fd.execute(ins)

    torch.testing.assert_close(outs[0], ins[1][:, :, 16:], rtol=0, atol=0)
    torch.testing.assert_close(
        outs[1],
        torch.where(
            torch.logical_or(ins[0] == 1, ins[2].unsqueeze(-1) == 1), 0, ins[1]
        ),
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
