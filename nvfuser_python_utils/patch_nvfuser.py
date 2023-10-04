import os


def patch_pytorch_nvfuser_binaries(torch_lib):
    nvfuser_lib = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "nvfuser", "lib"
    )
    import shutil

    for f_name in ["libnvfuser_codegen.so"]:
        target_lib_path = os.path.join(torch_lib, f_name)

        # Only overwrite target torch/lib/libnvfuser_codegen.so if it exists. (before nvfuser removal from torch @ torch <= 2.1)
        # Don't copy anything there if it doesn't exist. (after nvfuser removal from torch @ torch > 2.1)
        if os.path.isfile(target_lib_path):
            shutil.copyfile(
                os.path.join(nvfuser_lib, f_name),
                target_lib_path
            )


def remove_nvfuser_python_module(installed_nvfuser_dir):
    # only remove if installed nvfuser is in a different path
    if installed_nvfuser_dir != os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "nvfuser"
    ):
        import shutil

        shutil.rmtree(installed_nvfuser_dir)


def patch_installation():
    from importlib import util

    torch_dir = os.path.dirname(util.find_spec("torch").origin)
    torch_lib = os.path.join(torch_dir, "lib")

    installed_nvfuser_dir = os.path.join(os.path.dirname(torch_dir), "nvfuser")

    patch_pytorch_nvfuser_binaries(torch_lib)
    if os.path.exists(installed_nvfuser_dir):
        remove_nvfuser_python_module(installed_nvfuser_dir)


if __name__ == "__main__":
    patch_installation()
