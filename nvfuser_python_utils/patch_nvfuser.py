import os

_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def patch_pytorch_nvfuser_binaries(torch_lib):
    nvfuser_lib = os.path.join(_ROOT_DIR, "nvfuser", "lib")
    import shutil

    for f_name in ["libnvfuser_codegen.so"]:
        src_file = os.path.join(nvfuser_lib, f_name)
        tgt_file = os.path.join(torch_lib, f_name)
        if os.path.exists(src_file):
            print(f"Copy `{src_file}` -> `{tgt_file}`")
            shutil.copyfile(src_file, tgt_file)


def remove_nvfuser_python_module(installed_nvfuser_dir):
    # only remove if installed nvfuser is in a different path
    if installed_nvfuser_dir == os.path.join(_ROOT_DIR, "nvfuser"):
        print(
            f"nvFuser [{installed_nvfuser_dir}] is in among installed packages; no action required."
        )
        return
    import shutil

    print(f"Removing installed nvFuser at `{installed_nvfuser_dir}`")
    shutil.rmtree(installed_nvfuser_dir)


def verify_binary_installation():
    import nvfuser

    try:
        assert nvfuser._C._binary_verification() == "nvfuser_c_python_bindings"
    except (AttributeError, AssertionError) as err:
        logging.getLogger("nvfuser").error(
            "nvfuser probably loaded the wrong _C library from torch's legacy nvfuser submodule, "
            "try to reinstall nvfuser package AFTER torch, and run `patch-nvfuser` after installation"
        )
        raise err


def patch_installation():
    from importlib import util

    torch_dir = os.path.dirname(util.find_spec("torch").origin)
    torch_lib = os.path.join(torch_dir, "lib")

    installed_nvfuser_dir = os.path.join(os.path.dirname(torch_dir), "nvfuser")

    patch_pytorch_nvfuser_binaries(torch_lib)
    if os.path.exists(installed_nvfuser_dir):
        remove_nvfuser_python_module(installed_nvfuser_dir)

    verify_binary_installation()


if __name__ == "__main__":
    patch_installation()
