# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import subprocess
from pathlib import Path

UNKNOWN = "Unknown"
nvfuser_root = Path(__file__).parent.parent

# note that this root currently is still part of pytorch.
def get_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=nvfuser_root)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN

def get_version() -> str:
    sha = get_sha()
    version = open((nvfuser_root / "version.txt"), "r").read().strip() + "+git" + sha[:7]
    return version

def get_pytorch_cmake_prefix():
    import torch.utils
    return torch.utils.cmake_prefix_path

if __name__ == "__main__":
    version_file = nvfuser_root / "nvfuser" / "version.py"
    with open(version_file, "w") as f:
        f.write("_version_str = '{}'\n".format(get_version()))
