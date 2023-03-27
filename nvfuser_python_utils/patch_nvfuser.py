import os
import sys

def patch_pytorch_nvfuser_binaries():
    from subprocess import Popen, PIPE

    # TODO: exception handling for better error message
    # need to do this in a separate process so we are not going to delete nvfuser library while it's loaded by torch
    process_torch_lib = Popen(
        [
            "python",
            "-c",
            "import os; import torch; print(os.path.dirname(torch.__file__))",
        ],
        stdout=PIPE,
    )
    stdout_msg, error_msg = process_torch_lib.communicate()
    torch_lib = os.path.join(stdout_msg.decode("utf-8").rstrip("\n"), "lib")

    nvfuser_lib = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "nvfuser", "lib"
    )
    import shutil

    for f_name in ["libnvfuser_codegen.so"]:
        shutil.copyfile(
            os.path.join(nvfuser_lib, f_name),
            os.path.join(torch_lib, f_name),
        )

def patch_nvfuser_python_module():
    from subprocess import Popen, PIPE

    # TODO: exception handling for better error message
    process_torch_lib = Popen(
        [
            "python",
            "-c",
            "import os; import nvfuser; print(os.path.dirname(nvfuser.__file__))",
        ],
        stdout=PIPE,
    )
    stdout_msg, error_msg = process_torch_lib.communicate()
    installed_nvfuser_dir = stdout_msg.decode("utf-8").rstrip("\n")

    # only remove if installed nvfuser is in a different path
    if (installed_nvfuser_dir != os.path.join(os.path.dirname(os.path.dirname(__file__)), "nvfuser")):
        import shutil
        shutil.rmtree(installed_nvfuser_dir)

def patch_installation():
    patch_nvfuser_python_module()
    patch_pytorch_nvfuser_binaries()

if __name__ == "__main__":
    patch_installation()
