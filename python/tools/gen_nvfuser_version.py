# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import subprocess
import sys
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
        import os

        # assume the $NVFUSER_VERSION is in sha form
        if nvfuser_version := os.environ.get("NVFUSER_VERSION"):
            assert (
                len(nvfuser_version) < 11
            ), "The NVFUSER_VERSION should be in sha form"
            return nvfuser_version
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
            sys.executable,
            "-c",
            "import torch.utils; print(torch.utils.cmake_prefix_path)",
        ],
        stdout=PIPE,
    )
    stdout_msg, error_msg = process_torch_prefix.communicate()
    return stdout_msg.decode("utf-8").rstrip("\n")


def get_pytorch_use_distributed():
    from subprocess import Popen, PIPE

    # need to do this in a separate process so we are not going to delete nvfuser library while it's loaded by torch
    process_torch_prefix = Popen(
        [
            sys.executable,
            "-c",
            "import torch; print(torch._C._has_distributed())",
        ],
        stdout=PIPE,
    )
    stdout_msg, error_msg = process_torch_prefix.communicate()
    return stdout_msg.decode("utf-8").rstrip("\n")


if __name__ == "__main__":
    assert len(sys.argv) == 2
    assert sys.argv[1] == "nvfuser" or sys.argv[1] == "nvfuser_direct"
    python_module = sys.argv[1]
    version_file = nvfuser_root / python_module / "version.py"
    with open(version_file, "w") as f:
        f.write("_version_str = '{}'\n".format(get_version()))
