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
            shutil.move(target_lib_path, target_lib_path + ".old")
            shutil.copyfile(os.path.join(nvfuser_lib, f_name), target_lib_path)


def remove_nvfuser_python_module(installed_nvfuser_dir):
    # only remove if installed nvfuser is in a different path
    if installed_nvfuser_dir != os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "nvfuser"
    ):
        import shutil

        shutil.rmtree(installed_nvfuser_dir)


def get_torch_dirs():
    from importlib import util

    torch_dir = os.path.dirname(util.find_spec("torch").origin)
    torch_lib = os.path.join(torch_dir, "lib")

    return (torch_dir, torch_lib)


def patch_installation():
    torch_dir, torch_lib = get_torch_dirs()
    installed_nvfuser_dir = os.path.join(os.path.dirname(torch_dir), "nvfuser")

    patch_pytorch_nvfuser_binaries(torch_lib)
    if os.path.exists(installed_nvfuser_dir):
        remove_nvfuser_python_module(installed_nvfuser_dir)


def patch_installation_if_needed():
    import filecmp

    torch_dir, torch_lib = get_torch_dirs()

    installed_nvfuser_dir = os.path.join(os.path.dirname(torch_dir), "nvfuser")
    nvfuser_lib = os.path.join(installed_nvfuser_dir, "lib")

    f_name = "libnvfuser_codegen.so"

    torch_libnvfuser_codegen = os.path.join(torch_lib, f_name)
    nvfuser_libnvfuser_codegen = os.path.join(nvfuser_lib, f_name)

    if (
        os.path.isfile(torch_libnvfuser_codegen)
        and os.path.isfile(nvfuser_libnvfuser_codegen)
        and not filecmp.cmp(torch_libnvfuser_codegen, nvfuser_libnvfuser_codegen)
    ):
        print(
            f"nvfuser: found differences of {f_name} in {torch_lib} and {nvfuser_lib}. "
            "The library is now patched automatically in the first `import nvfuser` run. "
            f"The old {f_name} in {torch_lib} will be moved to {torch_lib}.old . "
            "This is not an error. No action is needed."
        )
        patch_installation()


if __name__ == "__main__":
    patch_installation()
