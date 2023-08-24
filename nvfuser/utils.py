import os


__all__ = [
    "cmake_prefix_path",
]


cmake_prefix_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "nvfuser",
    "share",
    "cmake",
    "nvfuser",
)
