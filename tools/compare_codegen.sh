#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# compare_codegen.sh - compare generated CUDA kernels between git commits
#
# This script is made to compare generated kernels when making changes to
# codegen. Invoking it without any arguments while checkout origin/main, as
# well as this commit. For each, it will build the project in release mode then
# invoke all tests and benchmarks, saving the generated cuda kernels to files.
# It will then diff these files and report which ones changed. The exit code is
# the number of cuda kernels that differ between commits.
#
# The -r option controls the git ref to compare against. The -o option is the
# name of the output directory.
#
# If the -- option is given, then instead of running all benchmarks and tests,
# we will use the rest of the command line as the test to run. For example, to
# compare the generated code for a single binary test, you could use:
#
#   tools/compare_codegen.sh -- build/nvfuser_tests --gtest_filter='*TestFoo*'
#
# In that case, the outputs will be placed in a subdirectory of the output
# directory for each commit labelled "custom_command_$LAUNCHTIME" where
# $LAUNCHTIME is a string representing the time this script was launched.

set -e
set -o pipefail

usage() {
  echo "Usage: $0 [-h] [-r origin/main] [-o codegen_comparison] [-- custom command to run]"
  echo -n "If given, custom command should only run a single executable. "
  echo "If multiple executables are run, kernels may be overwritten."
}

comparetoref=origin/main
outdir=codegen_comparison

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

trap "cleanup" INT TERM EXIT

run_test() {
    testdir=$1
    if [[ -d "$testdir/cuda" ]]
    then
        echo "Skipping since $testdir/cuda exists"
        return
    fi

    shift
    testcmd=$*

    mkdir -p "$testdir"
    echo "$testcmd" > "$testdir/command.txt"

    # Allow next command to fail
    set +e
    $testcmd | tee "$testdir/stdout-$(date +%Y%m%d_%H%M%S).log"
    echo $? > $testdir/returncode
    set -e
    mkdir -p "$testdir/cuda"
    movecudafiles "$testdir/cuda"
}


collect_kernels() {
    outdir=$1
    commit=$2

    git -c advice.detachedHead=false checkout "$commit"
    git submodule update --init --recursive
    currentcommit=$commit

    customcmddir=$outdir/$commit/custom_command_$launchtime

    binarytestdir=$outdir/$commit/binary_tests
    benchdir=$outdir/$commit/benchmarks
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
      if [[ -d "$binarytestdir/cuda" && -d "$benchdir/cuda" && -d "$pyfrontenddir/cuda" &&
          -d "$pyopsdir/cuda" && -d "$pyschedopsdir/cuda" && -d "$torchscriptdir/cuda" ]]
      then
          return
      fi
    fi

    # Build in Release mode
    python setup.py develop

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
      run_test "$pyopsdir" python -m pytest python_tests/pytest_ops.py -v -s --color=yes
      run_test "$pyschedopsdir" python -m pytest python_tests/test_schedule_ops.py -v -s --color=yes
      run_test "$pyfrontenddir" python -m pytest python_tests/test_python_frontend.py -v -s --color=yes
      run_test "$torchscriptdir" python -m pytest python_tests/test_torchscript.py -v -s --color=yes

      # binary tests
      run_test "$binarytestdir" build/nvfuser_tests --gtest_color=yes

      # benchmarks
      run_test "$benchdir" build/nvfuser_bench \
              --benchmark_repetitions=1 \
              --benchmark_min_time=0 \
              --benchmark_enable_random_interleaving=false \
              --benchmark_filter=NvFuserScheduler \
              --benchmark_color=true
    fi
}

collect_kernels "$outdir" "$origcommit"
collect_kernels "$outdir" "$comparecommit"

cleanup

# Print mismatching files. Note that logs are expected to differ since timings are included
diffs=$(diff -qr -x '*.log' "$outdir/$origcommit" "$outdir/$comparecommit")
echo "$diffs"

# Return number of mismatched cuda files. Success=0
num_mismatches=$(echo "$diffs" | wc -l)
exit "$num_mismatches"
