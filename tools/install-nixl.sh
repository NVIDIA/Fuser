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

# Locate shared library directories from the nixl pip packages.
# We avoid "import nixl" because the native extension may fail to load on
# headless CI runners without GPU/RDMA drivers.
#
# meson-python places bundled libs in  .nixl_cu*.mesonpy.libs/
# auditwheel places bundled libs in    nixl_cu*.libs/
# Both patterns are searched for nixl and nixl-cu* packages.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

FOUND_LIBNIXL=false

# Clone NIXL repo (shallow) for C headers.
git clone --depth 1 "$NIXL_REPO" "$NIXL_CLONE_DIR"

mkdir -p "$NIXL_PREFIX/include" "$NIXL_PREFIX/lib"

cp "$NIXL_CLONE_DIR"/src/api/cpp/*.h "$NIXL_PREFIX/include/"

# Symlink all shared libraries from every nixl-related libs directory so that
# transitive dependencies of libnixl.so (libserdes.so, libstream.so,
# libnixl_common.so, libetcd-cpp-api-core, etc.) are discoverable by the linker.
for libs_dir in "$SITE_PACKAGES"/.nixl*.mesonpy.libs "$SITE_PACKAGES"/nixl*.libs; do
    [ -d "$libs_dir" ] || continue
    echo "  Symlinking libs from $libs_dir"
    for so in "$libs_dir"/*.so*; do
        [ -e "$so" ] || continue
        ln -sf "$so" "$NIXL_PREFIX/lib/$(basename "$so")"
    done
done

# Fallback: search for any remaining nixl-related .so files anywhere in site-packages.
# Some transitive deps (e.g. libetcd-cpp-api-core) may be in package subdirectories.
for so in $(find "$SITE_PACKAGES" -maxdepth 3 -path "*/nixl*/*.so*" -type f 2>/dev/null); do
    name=$(basename "$so")
    [ -e "$NIXL_PREFIX/lib/$name" ] || ln -sf "$so" "$NIXL_PREFIX/lib/$name"
done

if [ ! -f "$NIXL_PREFIX/lib/libnixl.so" ]; then
    echo "Error: libnixl.so not found in any nixl libs directory under $SITE_PACKAGES"
    echo "Searched directories:"
    ls -d "$SITE_PACKAGES"/.nixl*.mesonpy.libs "$SITE_PACKAGES"/nixl*.libs 2>/dev/null || echo "  (none found)"
    exit 1
fi

rm -rf "$NIXL_CLONE_DIR"

echo "NIXL prefix ready at $NIXL_PREFIX"
echo "  include: $(ls "$NIXL_PREFIX/include/")"
echo "  lib:     $(ls -l "$NIXL_PREFIX/lib/")"
