#!/usr/bin/env bash

# Builds flatc and runs it on flatbuffer files

TOP_LEVEL=$(dirname $(dirname $(readlink -f ${BASH_SOURCE})))
# Determine build dir in typical location <top_level>/build/flatbuffers
BUILD_DIR=${TOP_LEVEL}/build/flatbuffers

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake $TOP_LEVEL/third_party/flatbuffers
ninja
cd $TOP_LEVEL/csrc/serde
$BUILD_DIR/flatc --cpp ./python_fusion_cache.fbs
