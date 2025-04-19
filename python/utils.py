# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

def check_env_flag_bool(name: str, default: str = "") -> bool:
    if name not in os.environ:
        return default
    return os.getenv(name).upper() in ["ON", "1", "YES", "TRUE", "Y"]

def parse_args():
    parser = argparse.ArgumentParser(description='NVFUSER build options', add_help=False)
    
    # Add arguments that don't go to setuptools
    parser.add_argument('--cmake-only', dest='cmake_only', action='store_true',
                        help='Only generate ./build directory with cmake setup')
    parser.add_argument('--no-python', dest='no_python', action='store_true',
                        help='Skips python API target libnvfuser.so')
    parser.add_argument('--no-test', dest='no_test', action='store_true',
                        help='Skips cpp tests test_nvfuser')
    parser.add_argument('--no-benchmark', dest='no_benchmark', action='store_true',
                        help='Skips benchmark target nvfuser_bench')
    parser.add_argument('--no-ninja', dest='no_ninja', action='store_true',
                        help='Use make instead of ninja for build')
    parser.add_argument('--build-with-ucc', dest='build_with_ucc', action='store_true',
                        help='Build nvfuser with UCC support')
    parser.add_argument('--explicit-error-check', dest='explicit_error_check', action='store_true',
                        help='Enable explicit error checking')
    parser.add_argument('--build-with-asan', dest='build_with_asan', action='store_true',
                        help='Build with Address Sanitizer')
    parser.add_argument('--build-without-distributed', dest='build_without_distributed', action='store_true',
                        help='Build nvfuser without multidevice support')
    parser.add_argument('--no-system-nvtx', dest='no_system_nvtx', action='store_true',
                        help='Disable system NVTX')
    parser.add_argument('--debug', dest='debug_mode', action='store_true',
                        help='Building nvfuser in debug mode')
    parser.add_argument('--debinfo', dest='debinfo_mode', action='store_true',
                        help='Building nvfuser in release mode with debug info')
    parser.add_argument('--build-dir', dest='build_dir', type=str, default="",
                        help='Specify in which directory to build nvfuser')
    parser.add_argument('--install-dir', dest='install_dir', type=str, default="",
                        help='Specify in which directory to install nvfuser')
    parser.add_argument('-install_requires', dest='install_requires', type=str,
                        help='Specify package required for installation')
    parser.add_argument('--extras_require', dest='extras_require', type=str,
                        help='Specify extra requirements')
    parser.add_argument('-version-tag', dest='version_tag', type=str,
                        help='Specify the tag for build nvfuser version')
    parser.add_argument('-wheel-name', dest='wheel_name', type=str, default="nvfuser",
                        help='Specify the wheel name')
    parser.add_argument('--cpp', dest='cpp_standard', type=int,
                        help='Specify the C++ standard to use', default=20)
    
    # Use parse_known_args to separate our arguments from setuptools arguments
    args, forward_args = parser.parse_known_args()
    return args, forward_args
