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
    version = (
        open((nvfuser_root / "version.txt"), "r").read().strip() + "+git" + sha[:7]
    )
    return version


def get_pytorch_cmake_prefix():
    from subprocess import Popen, PIPE

    # need to do this in a separate process so we are not going to delete nvfuser library while it's loaded by torch
    process_torch_prefix = Popen(
        [
            "python",
            "-c",
            "import torch.utils; print(torch.utils.cmake_prefix_path)",
        ],
        stdout=PIPE,
    )
    stdout_msg, error_msg = process_torch_prefix.communicate()
    return stdout_msg.decode("utf-8").rstrip("\n")


if __name__ == "__main__":
    version_file = nvfuser_root / "nvfuser" / "version.py"
    with open(version_file, "w") as f:
        f.write("_version_str = '{}'\n".format(get_version()))
