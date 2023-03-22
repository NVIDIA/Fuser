import os
import sys

# need to do this in a separate process so we are not going to delete nvfuser library while it's loaded by torch
def lookup_torch_dir():
    import torch

    return os.path.join(os.path.dirname(torch.__file__), "lib")


def patch_pytorch_nvfuser_binaries():
    from subprocess import Popen, PIPE

    # TODO: exception handling for better error message
    process_torch_lib = Popen(
        [
            "python",
            "-c",
            "from nvfuser_python_utils import lookup_torch_dir; print(lookup_torch_dir())",
        ],
        stdout=PIPE,
    )
    stdout_msg, error_msg = process_torch_lib.communicate()
    torch_lib = stdout_msg.decode("utf-8").rstrip("\n")

    nvfuser_lib = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "nvfuser", "lib"
    )
    import shutil

    for f_name in ["libnvfuser_codegen.so"]:
        shutil.copyfile(
            os.path.join(nvfuser_lib, f_name),
            os.path.join(torch_lib, f_name),
        )
