import subprocess
import os
import multiprocessing

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils.file_util import copy_file


ext_modules = []
ext_modules.append(Extensio(name=str("nvfuser._C"), sources=[]))

def get_cmake_bin():
  # TODO: double check cmake version here and retrieve later version if necessary
  return "cmake"

# make build directories
build_dir_name = "build"
cwd = os.path.dirname(os.path.abspath(__file__))
cmake_build_dir = os.path.join(cwd, "build")
os.makedirs(cmake_build_dir)

pytorch_cmake_config = '''-DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"'''

# generate cmake directory
subprocess.check_call("".join([get_cmake_bin(), pytorch_cmake_config, "-B", build_dir_name, "."]))

# build binary
max_jobs = str(multiprocessing.cpu_count())
subprocess.check_call("".join([get_cmake_bin(), "--build", build_dir_name, "-- -j", max_jobs))

# copy nvfuser pybind extension
src = os.path.join(cmake_build_dir, "libnvfuser.so")
dst = os.path.join(cwd, "nvfuser", "_C.cpython-310-x86_64-linux-gnu.so")

copy(src, dst)

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
