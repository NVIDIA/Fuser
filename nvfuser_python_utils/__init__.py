import os
from .patch_nvfuser import patch_installation

__all__ = ["patch_installation"]

cmake_prefix_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "nvfuser",
    "share",
    "cmake",
    "nvfuser",
)
