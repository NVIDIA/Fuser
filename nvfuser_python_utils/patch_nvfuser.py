import warnings


def patch_installation():
    warnings.warn(
        "`patch-nvfuser` is no longere needed after nvfuser==0.1.0. "
        "This binary will throw fatal error in the next minor release and be removed after that. ",
        stacklevel=2,
    )


if __name__ == "__main__":
    patch_installation()
