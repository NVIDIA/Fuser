import pytest
import torch
from nvfuser import FusionDefinition, DataType


def test_validate_precomputed_values():
    def compare() -> FusionDefinition:
        with FusionDefinition() as fd:
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
            )

            S1 = fd.define_scalar(None, dtype=DataType.Double)
            T2 = fd.ops.ge(T0, S1)
            fd.add_output(T2)
        return fd

    fd = compare()
    outs = fd.execute(
        [
            torch.randn((10,), dtype=torch.float32, device="cuda:0").as_strided(
                (2, 5), (5, 1)
            ),
            float("nan"),
        ]
    )
    # Cmoparing any number to NaN results in False.
    torch.testing.assert_close(outs[0].cpu(), torch.full((2, 5), False))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
