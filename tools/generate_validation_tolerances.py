# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
This script reproduces the computation used for generating the validation tolerances in csrc/validator_utils.h:
https://github.com/NVIDIA/Fuser/blob/892b7ac5646b3b873f13f2b2ea1db15fa9b9effb/csrc/validator_utils.h#L25-L46

The computation (randn + mul + sum) is repeated a large number of times.
The base tolerance is the max error seen when comparing the desired precision type with the next higher precision.
The final tolerances are computed by adding a safety factor of 3/4 to this base tolerance.

The output of this file is a .npy file corresponding to each dtype, containing a {size: safety_factor * base_tolerances} dict.

Note: The script may take a long time for a high number of num_iters, so it may be useful to only run the script for one dtype at a time. Modify `dtype_to_ref_dtypes` accordingly.

To run: python tools/generate_validation_tolerances.py
To load the files:
    ```
    import numpy as np
    tol_dict = np.load(file_name.npy, allow_pickle=True).item()
    ```
"""

import numpy as np
import torch
from datetime import datetime

sizes = [2**i for i in range(2, 22)]  # {4, 2097152}
num_iters = 10**6


def compute_max_error(size, dtype, ref_dtype):
    a, b = [torch.randn(size, dtype=dtype, device="cuda") for _ in range(2)]
    out = (a * b).sum()
    ref = (a.to(ref_dtype) * b.to(ref_dtype)).sum()
    max_error = out.sub(ref).abs().max().to(torch.double).item()
    return max_error


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA device is required."

    device = torch.cuda.get_device_name(torch.cuda.current_device())
    # Comparison is made against the next higher precision,
    # since it seems sufficient and avoids incurring extra memory.
    dtype_to_ref_dtypes = {
        torch.bfloat16: torch.float32,
        torch.float16: torch.float32,
        torch.float32: torch.float64,
    }

    # The original safety factor used is not known, but reverse-engineering estimates it between 3-5.
    safety_factor = 4

    for dtype, ref_dtype in dtype_to_ref_dtypes.items():
        tolerances = {}
        for size in sizes:
            print(f"Computing validation constant for size:{size}")
            base_tolerance = torch.finfo(torch.double).min
            for i in range(num_iters):
                base_tolerance = max(
                    base_tolerance, compute_max_error(size, dtype, ref_dtype)
                )
            tolerances[size] = safety_factor * base_tolerance
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        np.save(f"validation_consts_{dtype}_{device}_{date}.npy", tolerances)
        print(
            f"dtype:{dtype} device:{device} safety_factor:{safety_factor}\n{tolerances}\n"
        )
