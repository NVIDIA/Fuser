#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# compare_codegen.sh - compare generated CUDA kernels between git commits
#
# This script is made to compare generated kernels when making changes to
# codegen. Invoking it without any arguments will checkout origin/main, as well
# as this commit. For each, it will build the project in release mode then
# invoke all binary and python tests, saving the generated cuda kernels to .cu
# files. It will then diff these files and report which ones changed. The exit
# code is 1 if there are differences.
#
# The -r option controls the git ref to compare against. The -o option lets you
# specify the output directory.
#
# If the -- option is given then instead of running all tests, we will use the
# rest of the command line as the test to run. For example, to compare the
# generated code for a single binary test, you could use:
#
#   tools/compare_codegen.sh -- build/nvfuser_tests --gtest_filter='*TestFoo*'
#
# or to run all benchmarks you can use:
#
#   tools/compare_codegen.sh -- build/nvfuser_bench \
#       --benchmark_filter=NvFuserScheduler \
#       --benchmark_repetitions=1 \
#       --benchmark_min_time=0
#
# In those cases, the outputs will be placed in a subdirectory of the output
# directory for each commit labelled "custom_command_$LAUNCHTIME" where
# $LAUNCHTIME is a string representing the time this script was launched.
#
# By default, `python setup.py develop` is used to rebuild the project.
# You can also set environment variable CUSTOM_BUILD_COMMAND if your build
# is different.

set -e
set -o pipefail

usage() {
  echo "Usage: $0 [-h] [-r origin/main] [-o codegen_comparison] [-- custom command to run]"
  echo -n "If given, the custom command should only run a single executable. "
  echo "If multiple executables are run, kernel files may be overwritten."
}

# top-level directory of nvfuser repo
nvfuserdir=$(dirname $(dirname $(readlink -f $BASH_SOURCE)))

comparetoref=origin/main
outdir=$nvfuserdir/codegen_comparison

while getopts "r:o:h-" arg
do
  case $arg in
    r)
      comparetoref=$OPTARG
      ;;
    o)
      outdir=$OPTARG
      ;;
    h | ?)
      usage
      exit 1
      ;;
  esac
done
# getopts stops parsing if it sees "--". We can detect that case and record command
while [[ $# -gt 0 ]]
do
  if [[ "$1" == "--" ]]
  then
    hascustomcommand=1
    shift
    break
  fi
  shift
done
customcommand=$*

if [[ $(git status --porcelain --untracked-files=no) ]]
then
    echo "Must use git checkout in order to compare. Commit changes before running this script."
    exit 1
fi

# save current commit and current head so we can switch back to branch
currentcommit=$(git describe --always --long)
origcommit=$currentcommit
orighead=$(git symbolic-ref --short HEAD)

comparecommit=$(git describe --always --long "$comparetoref")

# record launch time to name custom command directories consistently
launchtime=$(date +%Y%m%d_%H%M%S)

movecudafiles() {
    find . -maxdepth 1 -name '__tmp_kernel*.cu' -exec mv '{}' "$1" \;
}

cleanup() {
    numkernels=$(find . -maxdepth 1 -name '__tmp_kernel*.cu' | wc -l)

    if (( numkernels > 0 ))
    then
        backupdir=$outdir/${currentcommit}-interrupted
        echo "Interrupted. Backing up $numkernels .cu files to $backupdir"
        mkdir -p "$backupdir"
        movecudafiles "$backupdir"
    fi

    git switch "$orighead"
    git submodule update --init --recursive
}

trap "cleanup" EXIT

run_test() {
    export testdir=$1
    if [[ -d "$testdir/cuda" ]]
    then
        echo "Skipping since $testdir/cuda exists"
        return
    fi

    shift
    testcmd=$*

    mkdir -p "$testdir"
    echo "$testcmd" > "$testdir/command"

    # Allow next command to fail
    set +e
    $testcmd | tee "$testdir/stdout-$(date +%Y%m%d_%H%M%S).log"
    echo $? > "$testdir/exitcode"
    set -e
    mkdir -p "$testdir/cuda"
    movecudafiles "$testdir/cuda"
}


collect_kernels() {
    outdir=$1
    commit=$2

    # Make sure we are doing a clean rebuild. Otherwise we might get linking error.
    python setup.py clean

    git -c advice.detachedHead=false checkout "$commit"
    git submodule update --init --recursive
    currentcommit=$commit

    customcmddir=$outdir/$commit/custom_command_$launchtime

    binarytestdir=$outdir/$commit/binary_tests
    pyfrontenddir=$outdir/$commit/python_frontend_tests
    pyopsdir=$outdir/$commit/python_ops_tests
    pyschedopsdir=$outdir/$commit/python_shedule_ops_tests
    torchscriptdir=$outdir/$commit/python_torchscript_tests

    # Test for output directories and return early if they exist. This
    # avoids rebuilds when we are changing code and comparing repeatedly to
    # the same earlier commit.
    if [[ $hascustomcommand ]]
    then
      if [[ -d "$customcmddir/cuda" ]]
      then
          return
      fi
    else
      if [[ -d "$binarytestdir/cuda" && -d "$pyfrontenddir/cuda" &&
          -d "$pyopsdir/cuda" && -d "$pyschedopsdir/cuda" &&
          -d "$torchscriptdir/cuda" ]]
      then
          return
      fi
    fi

    # Build in Release mode
    (
        cd $nvfuserdir
        CUSTOM_BUILD_COMMAND="${CUSTOM_BUILD_COMMAND:-python setup.py develop}"
        bash -c "${CUSTOM_BUILD_COMMAND}"
    )

    # Make tests reproducible
    export NVFUSER_TEST_RANDOM_SEED=0
    export NVFUSER_DISABLE=parallel_compile
    # run tests and benchmarks with cuda_to_file and dump output to files
    export NVFUSER_DUMP=cuda_to_file

    mkdir -p "$outdir/$commit"

    if [[ $hascustomcommand ]]
    then
      run_test "$customcmddir" $customcommand
    else
      # python tests
      # Using -s to disable capturing stdout. This is important as it will let us see which tests creates each .cu file
      run_test "$pyopsdir" python -m pytest $nvfuserdir/python_tests/pytest_ops.py -n 0 -v -s --color=yes
      run_test "$pyschedopsdir" python -m pytest $nvfuserdir/python_tests/test_schedule_ops.py -n 0 -v -s --color=yes
      run_test "$pyfrontenddir" python -m pytest $nvfuserdir/python_tests/test_python_frontend.py -n 0 -v -s --color=yes
      run_test "$torchscriptdir" python -m pytest $nvfuserdir/python_tests/test_torchscript.py -n 0 -v -s --color=yes

      # binary tests
      run_test "$binarytestdir" $nvfuserdir/build/nvfuser_tests --gtest_color=yes
    fi
}

collect_kernels "$outdir" "$origcommit"
collect_kernels "$outdir" "$comparecommit"

cleanup

# Print mismatching files. Note that logs are expected to differ since timings are included
set +e  # exit status of diff is 1 if there are any mismatches
echo -e "\n\nDIFF RESULT:\n"
diff -qr -x '*.log' "$outdir/$origcommit" "$outdir/$comparecommit" && echo "No difference found"

exit $?
