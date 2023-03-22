import os
import sys

def patch_pytorch_nvfuser_binaries():
    import torch

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    nvfuser_lib = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nvfuser", "lib")
    import shutil
    for f_name in ["libnvfuser_codegen.so"]:
        shutil.copyfile(
            os.path.join(nvfuser_lib, f_name),
            os.path.joint(torch_lib, f_name),
        )
