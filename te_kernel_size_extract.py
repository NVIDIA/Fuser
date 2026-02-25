#!/usr/bin/env python3

# Runs KernelSizeExtract.py on the Transformer Engine shared library,
# then filters the resulting JSON to only entries for SM architectures
# ending in 'a' or 'f' (e.g. sm_90a, sm_100a, sm_103a, sm_110a, sm_120a).
# Optionally resolves each unique kernel name to a GitHub permalink.

import subprocess
import sys
import json
import os
import re
import csv
import argparse
from collections import Counter
from pathlib import Path

KERNEL_SIZE_EXTRACT = "/nvidia/internal/kernelSizeExtract.py"
DEFAULT_TE_SO = "/opt/pytorch/TransformerEngine/build/cmake/libtransformer_engine.so"
DEFAULT_SOURCE_DIR = "/opt/pytorch/TransformerEngine"
FULL_JSON = "/tmp/te_kernel_sizes_full.json"
FILTERED_JSON = "/tmp/te_kernel_sizes_filtered.json"
OUTPUT_DIR = "/opt/pytorch/nvfuser"
GITHUB_REPO = "https://github.com/NVIDIA/TransformerEngine"


def get_repo_commit(source_dir):
    """Return the git commit hash for source_dir, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", source_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_github_link(repo, commit, rel_path, line_no):
    """Build a GitHub permalink for the given file path and line number."""
    return f"{repo}/blob/{commit}/{rel_path}#L{line_no}"


def base_function_name(kernel_name):
    """Strip template parameters and argument list, returning the bare qualified name.

    Handles (anonymous namespace) components which would otherwise be
    mistaken for an argument list by the '(' stop character.
    """
    # Temporarily replace anonymous namespace so '(' doesn't truncate early
    placeholder = "__anon__"
    name = kernel_name.replace("(anonymous namespace)", placeholder)
    match = re.match(r"^([^<(]+)", name)
    base = match.group(1).strip() if match else name
    # Restore anonymous namespace and strip any trailing '::'
    base = base.replace(placeholder, "(anonymous namespace)").rstrip(":").strip()
    return base


def _extract_name_from_mangled(mangled):
    """Parse an Itanium ABI mangled name (_ZN...) and return the bare function name.

    Handles anonymous-namespace kernels of the form
      _ZN<ns>...<NN>_GLOBAL__N__<hash><LL><function_name>I...
    by parsing the length-prefixed component sequence and returning the last
    non-_GLOBAL__N__ component before the template argument list.
    Returns None on parse failure.
    """
    if not mangled.startswith("_ZN"):
        return None
    rest = mangled[3:]  # skip '_ZN'
    last_name = None
    while rest:
        ch = rest[0]
        if ch in "EIv":  # end of nested name or start of template args
            break
        m = re.match(r"^(\d+)", rest)
        if not m:
            break
        n = int(m.group(1))
        rest = rest[m.end() :]
        if len(rest) < n:
            break
        component = rest[:n]
        rest = rest[n:]
        if not component.startswith("_GLOBAL__N__"):
            last_name = component
    return last_name


def find_kernel_source(base_name, source_dir):
    """Search source_dir for the __global__ definition of base_name.

    Returns a list of (relative_path, line_number) tuples for all matches.

    Handles common TE patterns:
      1. Same line:   __global__ void kernel_name(...)
      2. Split line:  __global__ void __launch_bounds__(N)
                          kernel_name(...)
      3. Launch bounds on preceding line:
                      __launch_bounds__(N) __global__
                          void kernel_name(...)
    Uses a sliding 4-line window: if the function name appears on a line and
    __global__ appears anywhere within the 3 preceding lines, it's a match.

    Also handles still-mangled names (starting with _Z) by parsing the Itanium
    ABI length-prefixed encoding to recover the bare function name.
    """
    # If the name is still mangled (c++filt couldn't decode it), parse it
    if base_name.startswith("_Z"):
        search_name = _extract_name_from_mangled(base_name)
        if not search_name:
            return []
    else:
        # Extract the bare function name (after last '::')
        search_name = base_name.split("::")[-1]
        if not search_name or search_name == "(anonymous namespace)":
            parts = [
                p for p in base_name.split("::") if p and p != "(anonymous namespace)"
            ]
            search_name = parts[-1] if parts else ""
        if not search_name:
            return []

    matches = []
    extensions = {".cu", ".cuh", ".h", ".hpp", ".cpp"}
    source_path = Path(source_dir)
    search_roots = [source_path / "transformer_engine", source_path / "common"]
    for root in search_roots:
        if not root.exists():
            continue
        for fpath in root.rglob("*"):
            if fpath.suffix not in extensions:
                continue
            try:
                lines = fpath.read_text(errors="replace").splitlines()
            except OSError:
                continue
            # Sliding window: keep last 3 lines to check for __global__
            window = []
            for i, line in enumerate(lines, 1):
                window.append(line)
                if len(window) > 4:
                    window.pop(0)
                # Function name must appear on current line
                if search_name not in line:
                    continue
                # __global__ must appear on this line or within the preceding 3 lines
                if any("__global__" in w for w in window):
                    rel = str(fpath.relative_to(source_path))
                    matches.append((rel, i))
    return matches


def main():
    parser = argparse.ArgumentParser(description="Extract and analyse TE kernel sizes")
    parser.add_argument(
        "--so",
        default=DEFAULT_TE_SO,
        help=f"Path to libtransformer_engine.so (default: {DEFAULT_TE_SO})",
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help=f"Path to TE source tree for resolving kernel locations (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip cuobjdump step and reuse existing JSON files",
    )
    args = parser.parse_args()

    te_so = args.so
    source_dir = args.source_dir

    if not args.skip_extract:
        # Step 1 & 2: Run KernelSizeExtract.py to generate the full JSON
        print(f"Running KernelSizeExtract.py on {te_so} ...")
        cmd = [sys.executable, KERNEL_SIZE_EXTRACT, "-i", te_so, "-o", FULL_JSON]
        result = subprocess.run(cmd, cwd="/tmp")
        if result.returncode != 0:
            print("ERROR: KernelSizeExtract.py failed.", file=sys.stderr)
            sys.exit(1)
        print(f"Full kernel JSON written to {FULL_JSON}")
    else:
        print(f"Skipping extraction, reusing {FULL_JSON}")

    # Step 3: Filter to SM architectures ending with 'a' or 'f'
    print("Filtering entries for SM architectures ending in 'a' or 'f' ...")
    with open(FULL_JSON, "r") as f:
        data = json.load(f)

    filtered = [e for e in data if re.search(r"[af]$", e.get("sm", ""))]

    with open(FILTERED_JSON, "w") as f:
        json.dump(filtered, f, indent=4)

    # Summary
    sm_counts = Counter(e["sm"] for e in filtered)
    print(f"\nFiltered {len(filtered)} kernels (out of {len(data)} total):")
    for sm, count in sorted(sm_counts.items()):
        print(f"  {sm}: {count} kernels")
    print(f"\nFiltered JSON written to {FILTERED_JSON}")

    # Step 4: Write one CSV per SM architecture
    print("\nGenerating per-SM CSVs ...")
    csv_paths = []
    for sm in sorted(sm_counts.keys()):
        entries = [e for e in filtered if e["sm"] == sm]
        csv_path = os.path.join(OUTPUT_DIR, f"te_kernels_{sm}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["kernel_name"])
            for e in sorted(entries, key=lambda x: -x["size"]):
                clean_name = re.sub(r"^void\s+", "", e["demangled_name"])
                writer.writerow([clean_name])
        csv_paths.append((sm, csv_path))
        print(f"  {csv_path}  ({len(entries)} kernels)")

    # Step 5: Deduplicate by base function name
    print("\nGenerating deduplicated CSVs ...")
    dedup_paths = []
    for sm, csv_path in csv_paths:
        counts = Counter()
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if row:
                    counts[base_function_name(row[0])] += 1
        dedup_path = os.path.join(OUTPUT_DIR, f"te_kernels_{sm}_unique.csv")
        with open(dedup_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["kernel_name", "count"])
            for name, count in sorted(counts.items(), key=lambda x: -x[1]):
                writer.writerow([name, count])
        dedup_paths.append((sm, dedup_path))
        print(f"  {dedup_path}  ({len(counts)} unique base names)")

    # Step 6: Resolve kernel names to GitHub permalinks
    if os.path.isdir(source_dir):
        commit = get_repo_commit(source_dir)
        if commit:
            print(
                f"\nResolving kernel source locations from {source_dir} (commit {commit[:12]}) ..."
            )
        else:
            print(
                f"\nResolving kernel source locations from {source_dir} (commit unknown) ..."
            )

        for sm, dedup_path in dedup_paths:
            rows = []
            with open(dedup_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    rows.append(row)

            resolved_path = os.path.join(OUTPUT_DIR, f"te_kernels_{sm}_resolved.csv")
            with open(resolved_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["kernel_name", "count", "github_link"])
                for row in rows:
                    name = row["kernel_name"]
                    count = row["count"]
                    hits = find_kernel_source(name, source_dir)
                    if hits:
                        for rel_path, line_no in hits:
                            link = (
                                make_github_link(GITHUB_REPO, commit, rel_path, line_no)
                                if commit
                                else f"{rel_path}#L{line_no}"
                            )
                            writer.writerow([name, count, link])
                    else:
                        writer.writerow([name, count, ""])
            print(f"  {resolved_path}")
    else:
        print(f"\nSource dir {source_dir!r} not found — skipping Step 6.")


if __name__ == "__main__":
    main()
