#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -e
set -o pipefail

if (( $# > 2 ))
then
    echo "Usage: $0 [ compare_ref=origin/main ] [ out_dir=codegen_comparison ]"
    exit 1
fi

comparetoref=${1:-origin/main}
comparecommit=$(git describe --always --long $comparetoref)
outdir=${2:-codegen_comparison}

if [[ $(git status --porcelain --untracked-files=no) ]]
then
    echo "Must checkout main in order to compare. Commit changes before running this script."
    exit 1
fi

# save current commit and current head so we can switch back to branch
currentcommit=$(git describe --always --long)
origcommit=$currentcommit
orighead=$(git symbolic-ref --short HEAD)

movecudafiles() {
    find . -maxdepth 1 -name '__tmp_kernel*.cu' -exec mv '{}' $1 \;
}

cleanup() {
    numkernels=$(find . -maxdepth 1 -name '__tmp_kernel*.cu' | wc -l)

    if (( $numkernels > 0 ))
    then
        backupdir=$outdir/${currentcommit}-interrupted
        echo "Interrupted. Backing up $numkernels .cu files to $backupdir"
        mkdir -p $backupdir
        movecudafiles $backupdir
    fi

    git switch $orighead
}

trap "cleanup" INT TERM EXIT

collect_kernels() {
    outdir=$1
    commit=$2

    git checkout $commit
    currentcommit=$commit

    testdir=$outdir/$commit/test
    benchdir=$outdir/$commit/bench

    # build in release mode so that everything runs a bit faster
    if [[ -d $testdir/cuda && -d $benchdir/cuda ]]
    then
        # Test for output directories and return early if they exist. This
        # avoids rebuilds when we are changing code and comparing repeatedly to
        # the same earlier commit.
        return
    fi

    # Build in Release mode
    python setup.py develop

    # Make tests reproducible
    export NVFUSER_TEST_RANDOM_SEED=0
    # run tests and benchmarks with cuda_to_file and dump output to files
    export NVFUSER_DUMP=cuda_to_file

    mkdir -p $outdir/$commit

    if [[ ! -d $testdir/cuda ]]
    then
        mkdir -p $testdir
        build/nvfuser_tests --gtest_color=yes | \
            tee $testdir/stdout-$(date +%Y%m%d_%H%M%S).log
        mkdir -p $testdir/cuda
        movecudafiles $testdir/cuda
    else
        echo "Skipping tests since $testdir/cuda exists"
    fi

    if [[ ! -d $benchdir/cuda ]]
    then
        mkdir -p $benchdir
        build/nvfuser_bench \
            --benchmark_repetitions=1 \
            --benchmark_min_time=0 \
            --benchmark_enable_random_interleaving=false \
            --benchmark_color=true \
         | tee $benchdir/stdout-$(date +%Y%m%d_%H%M%S).log
        mkdir -p $benchdir/cuda
        movecudafiles $benchdir/cuda
    else
        echo "Skipping benchmarks since $benchdir/cuda exists"
    fi
}

collect_kernels $outdir $origcommit
collect_kernels $outdir $comparecommit

cleanup

# Print mismatching files. Note that logs are expected to differ since timings are included
diffs=$(diff -qr -x '*.log' $outdir/$origcommit $outdir/$comparecommit)
echo $diffs

# Return number of mismatched cuda files. Success=0
num_mismatches=$(echo "$diffs" | wc -l)
exit $num_mismatches
