#!/bin/bash

# Install NIXL headers and libraries for Fuser CI.
#
# Two modes:
#   1. pip (default for GitHub Actions): install pre-built wheels, clone repo
#      for headers, symlink shared libs into NIXL_PREFIX.
#   2. source (default when CUDA toolkit is detected): build UCX with CUDA
#      transport and NIXL from source so the UCX backend can register VRAM.
#
# Environment variables:
#   NIXL_PREFIX       – install prefix (default: /tmp/nixl-prefix)
#   NIXL_BUILD_MODE   – "pip", "source", or "auto" (default: auto)
#   CUDA_HOME         – CUDA toolkit root (auto-detected from nvcc)
#
# Used by:
#   .github/workflows/build.yml   (GitHub Actions compilation check)
#   Blossom GPU CI build jobs      (needs runtime UCX+CUDA support)
#   tools/ci-local-build.sh        (local Docker reproduction)

set -e

NIXL_PREFIX="${NIXL_PREFIX:-/tmp/nixl-prefix}"
NIXL_BUILD_MODE="${NIXL_BUILD_MODE:-auto}"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"
NIXL_CLONE_DIR="/tmp/nixl-repo"
UCX_CLONE_DIR="/tmp/ucx-src"
UCX_INSTALL_DIR="/tmp/ucx-install"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

detect_cuda_home() {
  if [ -n "$CUDA_HOME" ]; then
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [ -d /usr/local/cuda ]; then
    CUDA_HOME=/usr/local/cuda
  fi
}

resolve_build_mode() {
  if [ "$NIXL_BUILD_MODE" = "auto" ]; then
    detect_cuda_home
    if [ -n "$CUDA_HOME" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
      echo "Auto-detected CUDA at $CUDA_HOME — using source build for UCX+CUDA support"
      NIXL_BUILD_MODE="source"
    else
      echo "No CUDA toolkit with nvcc found — using pip install (compile-only)"
      NIXL_BUILD_MODE="pip"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Mode: pip  (headers + pre-built .so, no runtime CUDA guarantee)
# ---------------------------------------------------------------------------

install_pip() {
  echo "=== Installing NIXL via pip ==="

  pip install --no-deps nixl nixl-cu12 || pip install --no-deps nixl

  SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

  git clone --depth 1 "$NIXL_REPO" "$NIXL_CLONE_DIR"
  mkdir -p "$NIXL_PREFIX/include" "$NIXL_PREFIX/lib"
  cp "$NIXL_CLONE_DIR"/src/api/cpp/*.h "$NIXL_PREFIX/include/"

  for libs_dir in "$SITE_PACKAGES"/.nixl*.mesonpy.libs "$SITE_PACKAGES"/nixl*.libs; do
    [ -d "$libs_dir" ] || continue
    echo "  Symlinking libs from $libs_dir"
    for so in "$libs_dir"/*.so*; do
      [ -e "$so" ] || continue
      ln -sf "$so" "$NIXL_PREFIX/lib/$(basename "$so")"
    done
  done

  for so in $(find "$SITE_PACKAGES" -maxdepth 3 -path "*/nixl*/*.so*" -type f 2>/dev/null); do
    name=$(basename "$so")
    [ -e "$NIXL_PREFIX/lib/$name" ] || ln -sf "$so" "$NIXL_PREFIX/lib/$name"
  done

  if [ ! -f "$NIXL_PREFIX/lib/libnixl.so" ]; then
    echo "Error: libnixl.so not found under $SITE_PACKAGES"
    exit 1
  fi

  rm -rf "$NIXL_CLONE_DIR"
}

# ---------------------------------------------------------------------------
# Mode: source  (UCX built with CUDA transport, NIXL built from source)
# ---------------------------------------------------------------------------

install_source() {
  echo "=== Building NIXL from source with UCX+CUDA ==="
  detect_cuda_home

  if [ -z "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME not set and nvcc not found"
    exit 1
  fi
  echo "  CUDA_HOME=$CUDA_HOME"

  # --- build dependencies ---------------------------------------------------
  apt-get update -qq 2>/dev/null || true
  apt-get install -y -qq libtool autoconf automake pkg-config \
      libibverbs-dev librdmacm-dev libnuma-dev 2>/dev/null || true
  pip install meson ninja 2>/dev/null || pip3 install meson ninja

  # --- UCX with CUDA --------------------------------------------------------
  echo "--- Building UCX with CUDA support ---"
  if [ -d "$UCX_CLONE_DIR" ]; then rm -rf "$UCX_CLONE_DIR"; fi
  git clone --depth 1 -b v1.18.x https://github.com/openucx/ucx.git "$UCX_CLONE_DIR"
  (
    cd "$UCX_CLONE_DIR"
    ./autogen.sh
    ./contrib/configure-release \
      --prefix="$UCX_INSTALL_DIR" \
      --with-cuda="$CUDA_HOME" \
      --enable-mt
    make -j"$(nproc)"
    make install
  )

  export PKG_CONFIG_PATH="$UCX_INSTALL_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export LD_LIBRARY_PATH="$UCX_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

  # --- NIXL from source -----------------------------------------------------
  echo "--- Building NIXL from source ---"
  if [ -d "$NIXL_CLONE_DIR" ]; then rm -rf "$NIXL_CLONE_DIR"; fi
  git clone --depth 1 "$NIXL_REPO" "$NIXL_CLONE_DIR"
  (
    cd "$NIXL_CLONE_DIR"

    CUDA_INC="$CUDA_HOME/include"
    CUDA_LIB="$CUDA_HOME/lib64"
    [ -d "$CUDA_LIB" ] || CUDA_LIB="$CUDA_HOME/lib"

    meson setup builddir \
      --prefix="$NIXL_PREFIX" \
      -Ducx_path="$UCX_INSTALL_DIR" \
      -Dcudapath_inc="$CUDA_INC" \
      -Dcudapath_lib="$CUDA_LIB" \
      -Dbuild_tests=false \
      -Dbuild_examples=false \
      -Dbuildtype=release
    meson compile -C builddir
    meson install -C builddir
  )

  # Copy API headers if not already installed by meson
  mkdir -p "$NIXL_PREFIX/include"
  cp -n "$NIXL_CLONE_DIR"/src/api/cpp/*.h "$NIXL_PREFIX/include/" 2>/dev/null || true

  # Ensure UCX libs are alongside NIXL libs so everything is on one rpath.
  # Also copy UCX transport plugins (libuct_cuda.so etc.) so they're discoverable.
  mkdir -p "$NIXL_PREFIX/lib/ucx"
  for so in "$UCX_INSTALL_DIR"/lib/*.so*; do
    [ -e "$so" ] || continue
    ln -sf "$so" "$NIXL_PREFIX/lib/$(basename "$so")"
  done
  for so in "$UCX_INSTALL_DIR"/lib/ucx/*.so*; do
    [ -e "$so" ] || continue
    ln -sf "$so" "$NIXL_PREFIX/lib/ucx/$(basename "$so")"
  done

  rm -rf "$NIXL_CLONE_DIR" "$UCX_CLONE_DIR"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

resolve_build_mode
echo "NIXL build mode: $NIXL_BUILD_MODE"

case "$NIXL_BUILD_MODE" in
  pip)    install_pip    ;;
  source) install_source ;;
  *)
    echo "Error: unknown NIXL_BUILD_MODE=$NIXL_BUILD_MODE (expected pip, source, or auto)"
    exit 1
    ;;
esac

# Ensure LD_LIBRARY_PATH includes NIXL_PREFIX/lib for runtime
export LD_LIBRARY_PATH="$NIXL_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo ""
echo "NIXL prefix ready at $NIXL_PREFIX"
echo "  include: $(ls "$NIXL_PREFIX/include/" 2>/dev/null || echo '(empty)')"
echo "  lib:     $(ls "$NIXL_PREFIX/lib/" 2>/dev/null || echo '(empty)')"
echo ""
echo "Remember to set:"
echo "  export LD_LIBRARY_PATH=$NIXL_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  export UCX_MODULE_DIR=$NIXL_PREFIX/lib/ucx  (if UCX was built from source)"
