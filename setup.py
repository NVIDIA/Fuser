import subprocess
import os
import multiprocessing

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils.file_util import copy_file

# NOTE: do we need ext_modules here at all? we are not building anything.
ext_modules = []
ext_modules.append(Extension(name=str("nvfuser._C"), sources=[]))

def get_cmake_bin():
  # TODO: double check cmake version here and retrieve later version if necessary
  return "cmake"

# make build directories
build_dir_name = "build"
cwd = os.path.dirname(os.path.abspath(__file__))
cmake_build_dir = os.path.join(cwd, "build")
if not os.path.exists(cmake_build_dir):
  os.makedirs(cmake_build_dir)

import torch.utils
pytorch_cmake_config = "-DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path

# generate cmake directory
cmd_str = [get_cmake_bin(), pytorch_cmake_config, "-B", build_dir_name, "."]
subprocess.check_call(cmd_str)

# build binary
max_jobs = str(multiprocessing.cpu_count())
cmd_str = [get_cmake_bin(), "--build", build_dir_name, "--", "-j", max_jobs]
subprocess.check_call(cmd_str)

# copy nvfuser pybind extension
src = os.path.join(cmake_build_dir, "libnvfuser.so")
dst = os.path.join(cwd, "nvfuser", "_C.cpython-310-x86_64-linux-gnu.so")

copy_file(src, dst)

#nvfuser_package_data = [
#  'share/cmake/nvfuser/*.cmake',
#  'include/nvfuser/*.h',
#  'include/nvfuser/kernel_db/*.h',
#  'include/nvfuser/multidevice/*.h',
#  'include/nvfuser/ops/*.h',
#  'include/nvfuser/python_frontend/*.h',
#  'include/nvfuser/scheduler/*.h',
#]

setup(
  name="nvfuser",
  # query nvfuser version
  version="0.0.1",
  description="A Fusion Code Generator for NVIDIA GPUs (commonly known as 'nvFuser')",
  packages=["nvfuser"],
  ext_modules=ext_modules,
  license_files=("LICENSE",),
)
