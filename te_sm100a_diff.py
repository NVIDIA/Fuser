#!/usr/bin/env python3

# Variant of te_kernel_size_extract.py focused on sm_100 vs sm_100a.
#
# Steps:
#   1. Run KernelSizeExtract.py on a libtransformer_engine.so
#   2. Filter to sm_100 and sm_100a entries only
#   3. Write per-SM CSVs (all kernels, sorted by size descending)
#   4. Deduplicate each SM by base function name -> *_unique.csv
#   5. Compute the set difference: unique base names in sm_100a NOT in sm_100
#      -> te_kernels_sm_100a_only.csv
#   6. Resolve those sm_100a-only names to GitHub permalinks
#      -> te_kernels_sm_100a_only_resolved.csv

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
OUTPUT_DIR = "/opt/pytorch/nvfuser"
GITHUB_REPO = "https://github.com/NVIDIA/TransformerEngine"

TARGET_SMS = {"sm_100", "sm_100a"}


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
    placeholder = "__anon__"
    name = kernel_name.replace("(anonymous namespace)", placeholder)
    match = re.match(r"^([^<(]+)", name)
    base = match.group(1).strip() if match else name
    base = base.replace(placeholder, "(anonymous namespace)").rstrip(":").strip()
    return base


def _extract_name_from_mangled(mangled):
    """Parse an Itanium ABI mangled name (_ZN...) and return the bare function name."""
    if not mangled.startswith("_ZN"):
        return None
    rest = mangled[3:]
    last_name = None
    while rest:
        ch = rest[0]
        if ch in "EIv":
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
    Uses a sliding 4-line window: if the function name appears on a line and
    __global__ appears anywhere within the 3 preceding lines, it's a match.
    """
    if base_name.startswith("_Z"):
        search_name = _extract_name_from_mangled(base_name)
        if not search_name:
            return []
    else:
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
            window = []
            for i, line in enumerate(lines, 1):
                window.append(line)
                if len(window) > 4:
                    window.pop(0)
                if search_name not in line:
                    continue
                if any("__global__" in w for w in window):
                    rel = str(fpath.relative_to(source_path))
                    matches.append((rel, i))
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Compare sm_100 vs sm_100a kernels and find sm_100a-only entries"
    )
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
        help="Skip cuobjdump step and reuse existing full JSON",
    )
    args = parser.parse_args()

    te_so = args.so
    source_dir = args.source_dir

    # Step 1 & 2: Run KernelSizeExtract.py
    if not args.skip_extract:
        print(f"Running KernelSizeExtract.py on {te_so} ...")
        cmd = [sys.executable, KERNEL_SIZE_EXTRACT, "-i", te_so, "-o", FULL_JSON]
        result = subprocess.run(cmd, cwd="/tmp")
        if result.returncode != 0:
            print("ERROR: KernelSizeExtract.py failed.", file=sys.stderr)
            sys.exit(1)
        print(f"Full kernel JSON written to {FULL_JSON}")
    else:
        print(f"Skipping extraction, reusing {FULL_JSON}")

    # Step 3: Filter to sm_100 and sm_100a only
    print(f"\nFiltering to {TARGET_SMS} ...")
    with open(FULL_JSON, "r") as f:
        data = json.load(f)

    filtered = [e for e in data if e.get("sm") in TARGET_SMS]

    sm_counts = Counter(e["sm"] for e in filtered)
    print(f"Found {len(filtered)} kernels (out of {len(data)} total):")
    for sm, count in sorted(sm_counts.items()):
        print(f"  {sm}: {count} kernels")

    # Step 4: Write per-SM CSVs (all kernels, sorted by size desc)
    print("\nGenerating per-SM CSVs ...")
    csv_paths = {}
    for sm in sorted(TARGET_SMS):
        entries = [e for e in filtered if e["sm"] == sm]
        if not entries:
            print(f"  (no entries for {sm})")
            continue
        csv_path = os.path.join(OUTPUT_DIR, f"te_kernels_{sm}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["kernel_name", "size"])
            for e in sorted(entries, key=lambda x: -x["size"]):
                clean_name = re.sub(r"^void\s+", "", e["demangled_name"])
                writer.writerow([clean_name, e["size"]])
        csv_paths[sm] = csv_path
        print(f"  {csv_path}  ({len(entries)} kernels)")

    # Step 5: Deduplicate by base function name
    print("\nGenerating deduplicated CSVs ...")
    dedup_names = {}  # sm -> set of base names
    dedup_paths = {}
    for sm, csv_path in csv_paths.items():
        counts = Counter()
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                counts[base_function_name(row["kernel_name"])] += 1
        dedup_names[sm] = set(counts.keys())
        dedup_path = os.path.join(OUTPUT_DIR, f"te_kernels_{sm}_unique.csv")
        with open(dedup_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["kernel_name", "count"])
            for name, count in sorted(counts.items(), key=lambda x: -x[1]):
                writer.writerow([name, count])
        dedup_paths[sm] = dedup_path
        print(f"  {dedup_path}  ({len(counts)} unique base names)")

    # Step 5b: Compute sm_100a-only set (present in sm_100a but not sm_100)
    names_100a = dedup_names.get("sm_100a", set())
    names_100 = dedup_names.get("sm_100", set())
    only_in_100a = sorted(names_100a - names_100)

    only_path = os.path.join(OUTPUT_DIR, "te_kernels_sm_100a_only.csv")
    with open(only_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["kernel_name"])
        for name in only_in_100a:
            writer.writerow([name])
    print(f"\nKernels in sm_100a but NOT in sm_100: {len(only_in_100a)}")
    print(f"  {only_path}")

    if not only_in_100a:
        print("No sm_100a-only kernels found; skipping source resolution.")
        return

    # Step 6: Resolve sm_100a-only names to GitHub permalinks
    if os.path.isdir(source_dir):
        commit = get_repo_commit(source_dir)
        if commit:
            print(
                f"\nResolving source locations from {source_dir} (commit {commit[:12]}) ..."
            )
        else:
            print(
                f"\nResolving source locations from {source_dir} (commit unknown) ..."
            )

        resolved_path = os.path.join(OUTPUT_DIR, "te_kernels_sm_100a_only_resolved.csv")
        with open(resolved_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["kernel_name", "github_link"])
            for name in only_in_100a:
                hits = find_kernel_source(name, source_dir)
                if hits:
                    for rel_path, line_no in hits:
                        link = (
                            make_github_link(GITHUB_REPO, commit, rel_path, line_no)
                            if commit
                            else f"{rel_path}#L{line_no}"
                        )
                        writer.writerow([name, link])
                else:
                    writer.writerow([name, ""])
        print(f"  {resolved_path}")
    else:
        print(f"\nSource dir {source_dir!r} not found — skipping source resolution.")


if __name__ == "__main__":
    main()
