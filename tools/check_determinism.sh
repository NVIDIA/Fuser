#!/bin/bash

set -e
set -o pipefail

usage() {
  echo "Usage: $0 [-h] [-n NUMREPS=10] -- command to run]"
}


while getopts "n:h" arg
do
  case $arg in
    n)
      NUMREPS=$OPTARG
      shift
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
    shift
    break
  fi
  shift
done
CMD=$*

export NVFUSER_DUMP=cuda_to_file

KERNELDIR=$(mktemp -d)

cleanup() {
    rm -rf "$KERNELDIR"
}

trap "cleanup" EXIT

FIRSTREPDIR="$KERNELDIR/1"

retval=0
for rep in $(seq 1 "$NUMREPS")
do
    NUMEXISTINGCUFILES=$(find . -maxdepth 1 -name \*.cu | wc -l)
    if [[ $NUMEXISTINGCUFILES -ne 0 ]]
    then
        KERNELBACKUPDIR=./check_determinism-kernelbackup$(date +%Y%m%d.%H%M%S)
        echo "Backing up $NUMEXISTINGCUFILES existing .cu files to $KERNELBACKUPDIR"
        mkdir -p "$KERNELBACKUPDIR"
        mv ./*.cu "$KERNELBACKUPDIR"
    fi
    # $CMD does not need to succeed for us to analyze it
    set +e
    $CMD
    set -e

    REPDIR="$KERNELDIR/$rep"
    mkdir -p "$REPDIR"
    mv ./*.cu "$REPDIR/"

    NUMFIRST=$(find "$FIRSTREPDIR" -name \*.cu | wc -l)
    NUMREP=$(find "$REPDIR" -name \*.cu | wc -l)
    if [[ $NUMREP -ne $NUMFIRST ]]
    then
        echo "Created $NUMFIRST kernels on first repetition and $NUMREP on repetition $rep"
        retval=1
    fi
    for newkernel in "$REPDIR"/*.cu
    do
        basename=$(basename "$newkernel")
        firstkernel="$FIRSTREPDIR/$basename"
        if [ ! -f "$firstkernel" ]
        then
            echo "Kernel file $newkernel in repetition $rep does not exist in first repetition"
            retval=1
            continue
        fi
        set +e
        diff "$firstkernel" "$newkernel"
        diffstatus=$?
        set -e
        if [[ $diffstatus -ne 0 ]]
        then
            printf 'Diff of %s from rep 1 to rep %d (above) is non-zero\n' "$basename" "$rep"
            retval=1
            continue
        fi
    done
    if [[ $retval -ne 0 ]]
    then
        # Stop repetitions after first failure
        break
    else
        echo "Generated kernels all match"
    fi
done

exit $retval
