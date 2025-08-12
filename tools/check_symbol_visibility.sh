#!/bin/bash

# CI Script to check symbol visibility for nvfuser Python bindings
# Ensures all undefined symbols in Python modules are properly exported
# from the shared libraries they depend on.
#
# Usage: ./tools/check_symbol_visibility.sh
# Exit codes:
#   0: All symbols properly exported
#   1: Missing symbols found or other errors

set -e  # Exit on any error

# Get script directory and project root
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/python/build"
PYTHON_DIR="$PROJECT_ROOT/python"

# Create temporary directory for symbol files
TEMP_DIR=$(mktemp -d)

echo "=== nvfuser Symbol Visibility Check ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "Temp directory: $TEMP_DIR"
echo ""

# Function to check if file exists
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        return 1
    fi
}

# Function to find Python extension files
find_python_extensions() {
    local python_dir="$1"
    local pattern="$2"
    find "$python_dir" -name "$pattern" -type f 2>/dev/null || true
}

# 1. Check if required libraries exist
echo "Step 1: Checking required libraries..."
NVFUSER_CODEGEN_LIB="$BUILD_DIR/libnvfuser_codegen.so"
CUTLASS_LIB="$BUILD_DIR/libnvf_cutlass.so"

check_file_exists "$NVFUSER_CODEGEN_LIB" || exit 1

# Find Python extension files
NVFUSER_EXT=$(find_python_extensions "$PYTHON_DIR/nvfuser" "_C*.so")
NVFUSER_DIRECT_EXT=$(find_python_extensions "$PYTHON_DIR/nvfuser_direct" "_C_DIRECT*.so")

if [ -z "$NVFUSER_EXT" ]; then
    echo "ERROR: nvfuser Python extension (_C*.so) not found in $PYTHON_DIR/nvfuser"
    exit 1
fi

if [ -z "$NVFUSER_DIRECT_EXT" ]; then
    echo "ERROR: nvfuser_direct Python extension (_C_DIRECT*.so) not found in $PYTHON_DIR/nvfuser_direct"
    exit 1
fi

echo "Found nvfuser extension: $NVFUSER_EXT"
echo "Found nvfuser_direct extension: $NVFUSER_DIRECT_EXT"
echo ""

# 2. Get exported symbols from shared libraries
echo "Step 2: Getting exported nvfuser symbols from shared libraries..."

# Get symbols from main library
nm -D "$NVFUSER_CODEGEN_LIB" | grep nvfuser | awk '{print $3}' | sort > "$TEMP_DIR/nvfuser_codegen_symbols.txt"
CODEGEN_SYMBOL_COUNT=$(wc -l < "$TEMP_DIR/nvfuser_codegen_symbols.txt")
echo "Found $CODEGEN_SYMBOL_COUNT exported symbols from libnvfuser_codegen.so"

# Get symbols from cutlass library (if it exists)
if [ -f "$CUTLASS_LIB" ]; then
    nm -D "$CUTLASS_LIB" | grep nvfuser | awk '{print $3}' | sort > "$TEMP_DIR/nvf_cutlass_symbols.txt"
    CUTLASS_SYMBOL_COUNT=$(wc -l < "$TEMP_DIR/nvf_cutlass_symbols.txt")
    echo "Found $CUTLASS_SYMBOL_COUNT exported symbols from libnvf_cutlass.so"
    # Combine symbols from both libraries
    cat "$TEMP_DIR/nvfuser_codegen_symbols.txt" "$TEMP_DIR/nvf_cutlass_symbols.txt" | sort -u > "$TEMP_DIR/exported_symbols.txt"
else
    echo "libnvf_cutlass.so not found, using only libnvfuser_codegen.so symbols"
    cp "$TEMP_DIR/nvfuser_codegen_symbols.txt" "$TEMP_DIR/exported_symbols.txt"
fi

TOTAL_EXPORTED=$(wc -l < "$TEMP_DIR/exported_symbols.txt")
echo "Total unique exported symbols: $TOTAL_EXPORTED"
echo ""

# 3. Check symbols for each Python extension
check_extension_symbols() {
    local ext_file="$1"
    local ext_name="$2"

    echo "=== Checking $ext_name ==="

    # Get undefined nvfuser symbols from this extension
    local undefined_file="$TEMP_DIR/${ext_name}_undefined_symbols.txt"
    nm -u "$ext_file" | grep nvfuser | awk '{print $2}' | sort > "$undefined_file"
    local undefined_count=$(wc -l < "$undefined_file")
    echo "Found $undefined_count undefined nvfuser symbols in $ext_name"

    # Find missing symbols
    local missing_file="$TEMP_DIR/${ext_name}_missing_symbols.txt"
    comm -23 "$undefined_file" "$TEMP_DIR/exported_symbols.txt" > "$missing_file"
    local missing_count=$(wc -l < "$missing_file")

    if [ "$missing_count" -gt 0 ]; then
        echo "ERROR: Found $missing_count missing symbols in $ext_name:"
        echo ""
        # Show demangled names for readability
        while IFS= read -r symbol; do
            demangled=$(echo "$symbol" | c++filt 2>/dev/null || echo "$symbol")
            echo "  $demangled"
        done < "$missing_file"
        echo ""
        return 1
    else
        echo "SUCCESS: All undefined symbols in $ext_name are properly exported by libnvfuser_codegen.so"
        if [ -f "$CUTLASS_LIB" ]; then
            echo "         and libnvf_cutlass.so"
        fi
        echo ""
        return 0
    fi
}

# Check both extensions
NVFUSER_OK=0
NVFUSER_DIRECT_OK=0

check_extension_symbols "$NVFUSER_EXT" "nvfuser" || NVFUSER_OK=1
check_extension_symbols "$NVFUSER_DIRECT_EXT" "nvfuser_direct" || NVFUSER_DIRECT_OK=1

# 4. Final results
echo "=== FINAL RESULTS ==="
if [ $NVFUSER_OK -eq 0 ] && [ $NVFUSER_DIRECT_OK -eq 0 ]; then
    echo "✅ SUCCESS: All Python extensions have properly exported symbols"
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    exit 0
else
    echo "❌ FAILURE: Missing symbols detected"
    echo ""
    if [ $NVFUSER_OK -ne 0 ]; then
        echo "- nvfuser extension has missing symbols (see $TEMP_DIR/nvfuser_missing_symbols.txt)"
    fi
    if [ $NVFUSER_DIRECT_OK -ne 0 ]; then
        echo "- nvfuser_direct extension has missing symbols (see $TEMP_DIR/nvfuser_direct_missing_symbols.txt)"
    fi
    echo ""
    echo "This indicates that some symbols need NVF_API annotations or are not being"
    echo "exported from the shared libraries. Check the missing symbols above and add"
    echo "appropriate visibility annotations."
    echo ""
    echo "Debug files preserved in: $TEMP_DIR"
    echo "- nvfuser_codegen_symbols.txt: Exported symbols from libnvfuser_codegen.so"
    if [ -f "$CUTLASS_LIB" ]; then
        echo "- nvf_cutlass_symbols.txt: Exported symbols from libnvf_cutlass.so"
    fi
    echo "- exported_symbols.txt: Combined exported symbols from all libraries"
    echo "- nvfuser_undefined_symbols.txt: Undefined symbols from nvfuser extension"
    echo "- nvfuser_direct_undefined_symbols.txt: Undefined symbols from nvfuser_direct extension"
    if [ $NVFUSER_OK -ne 0 ]; then
        echo "- nvfuser_missing_symbols.txt: Missing symbols from nvfuser extension"
    fi
    if [ $NVFUSER_DIRECT_OK -ne 0 ]; then
        echo "- nvfuser_direct_missing_symbols.txt: Missing symbols from nvfuser_direct extension"
    fi
    echo ""
    echo "Remember to clean up manually: rm -rf $TEMP_DIR"
    exit 1
fi
