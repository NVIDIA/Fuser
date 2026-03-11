#!/bin/bash

# Install NIXL headers and libraries for Fuser CI compilation.
#
# The pip wheel provides libnixl.so (in a meson-python internal directory)
# but not the C development headers. We clone the NIXL repo to get headers
# and create a NIXL_PREFIX directory that handle_nixl.cmake can discover.
#
# Used by: .github/workflows/build.yml (GitHub Actions compilation check)
# For Blossom GPU CI: NIXL should be pre-installed in the CI Docker image
# (see dev/Dockerfile for reference), or this script can be run as a
# pre-build step if the runner has network access.

set -e

NIXL_PREFIX="${NIXL_PREFIX:-/tmp/nixl-prefix}"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
NIXL_CLONE_DIR="/tmp/nixl-repo"

# Use --no-deps to avoid pulling in nixl-cu12's torch/numpy dependencies,
# which would conflict with the torch nightly already installed by
# pip-install-things.sh (this script must run AFTER pip-install-things.sh).
pip install --no-deps nixl nixl-cu12

# Locate the mesonpy libs directory where libnixl.so lives.
# We avoid "import nixl" because the native extension may fail to load on
# headless CI runners without GPU/RDMA drivers.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
MESONPY_LIBS=$(find "$SITE_PACKAGES" -maxdepth 1 -name ".nixl_cu*.mesonpy.libs" -type d | head -1)

if [ -z "$MESONPY_LIBS" ] || [ ! -d "$MESONPY_LIBS" ]; then
    echo "Error: nixl mesonpy libs directory not found in $SITE_PACKAGES"
    exit 1
fi

if [ ! -f "$MESONPY_LIBS/libnixl.so" ]; then
    echo "Error: libnixl.so not found in $MESONPY_LIBS"
    exit 1
fi

# Clone NIXL repo (shallow) for C headers.
git clone --depth 1 "$NIXL_REPO" "$NIXL_CLONE_DIR"

mkdir -p "$NIXL_PREFIX/include" "$NIXL_PREFIX/lib"

cp "$NIXL_CLONE_DIR"/src/api/cpp/*.h "$NIXL_PREFIX/include/"

ln -sf "$MESONPY_LIBS/libnixl.so" "$NIXL_PREFIX/lib/libnixl.so"
if [ -f "$MESONPY_LIBS/libnixl_build.so" ]; then
    ln -sf "$MESONPY_LIBS/libnixl_build.so" "$NIXL_PREFIX/lib/libnixl_build.so"
fi

rm -rf "$NIXL_CLONE_DIR"

echo "NIXL prefix ready at $NIXL_PREFIX"
echo "  include: $(ls "$NIXL_PREFIX/include/")"
echo "  lib:     $(ls -l "$NIXL_PREFIX/lib/")"
