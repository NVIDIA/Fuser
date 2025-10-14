#!/usr/bin/env python3
"""
Generate doc/index/type_index.md by scanning csrc for class/struct/enum
definitions. Handles macro/attribute tokens between the keyword and name and
adds line anchors to each link. Produces two sections: Types and Enums.

Usage:
  Run from anywhere. Assumes repository repo_root is three directories up from here.
"""
import os
import re
import sys
from typing import Dict, Tuple, DefaultDict, List


def main() -> None:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    csrc = os.path.join(repo_root, "csrc")
    doc_index = os.path.join(repo_root, "doc", "index")
    out_path = os.path.join(doc_index, "type_index.md")

    ident = r"[A-Za-z_][A-Za-z0-9_]*"
    # Allow common macro/attribute tokens between class/struct and the real name
    macro = r"(?:NVF_API|NVF_HOST_DEVICE|NVF_API_T|__attribute__\s*\(.*?\)|__declspec\s*\(.*?\))"
    class_struct_re = re.compile(
        r"^\s*(class|struct)\s+(?:" + macro + r"\s+)*(" + ident + r")\b[^;{]*\{"
    )
    enum_re = re.compile(r"^\s*enum\s+(?:class|struct)?\s*(" + ident + r")\s*\{")

    # name, rel_path -> min line
    class_min: Dict[Tuple[str, str], int] = {}
    enum_min: Dict[Tuple[str, str], int] = {}

    for dirpath, _, filenames in os.walk(csrc):
        for fn in filenames:
            if not fn.endswith((".h", ".hpp", ".hh", ".cuh", ".cpp")):
                continue
            full = os.path.join(dirpath, fn)
            if full.endswith("csrc/serde/fusion_cache_generated.h"):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        m = class_struct_re.match(line)
                        if m:
                            name = m.group(2)
                            rel = os.path.relpath(full, doc_index)
                            key = (name, rel)
                            class_min[key] = min(i, class_min.get(key, i))
                            continue
                        m2 = enum_re.match(line)
                        if m2:
                            name = m2.group(1)
                            rel = os.path.relpath(full, doc_index)
                            key = (name, rel)
                            enum_min[key] = min(i, enum_min.get(key, i))
            except Exception:
                # Best-effort
                pass

    classes_sorted = sorted(class_min.items(), key=lambda x: (x[0][0].lower(), x[0][1]))
    enums_sorted = sorted(enum_min.items(), key=lambda x: (x[0][0].lower(), x[0][1]))

    # Collect base classes using libclang and compile_commands.json (required)
    def collect_bases() -> Dict[str, List[str]]:
        try:
            import json
            from clang.cindex import Index, TranslationUnit, CursorKind, Config
        except Exception:
            return {}
        # Ensure libclang is loaded if present in standard location
        try:
            if not Config.loaded:
                candidate = "/lib/x86_64-linux-gnu/libclang-cpp.so.18.1"
                if os.path.exists(candidate):
                    try:
                        Config.set_library_file(candidate)
                    except Exception:
                        pass
        except Exception:
            pass
        cc_path = os.path.join(repo_root, "python", "build", "compile_commands.json")
        if not os.path.exists(cc_path):
            return {}
        try:
            with open(cc_path, "r", encoding="utf-8") as f:
                db = json.load(f)
        except Exception:
            return {}
        # Build a map of source file -> args and directory -> args
        file_to_args: Dict[str, List[str]] = {}
        dir_to_args: Dict[str, List[str]] = {}
        for entry in db:
            file_path = os.path.abspath(entry.get("file") or entry.get("filename", ""))
            if not file_path:
                continue
            args: List[str] = []
            if "arguments" in entry and isinstance(entry["arguments"], list):
                args = [a for a in entry["arguments"][1:] if a]
            elif "command" in entry and isinstance(entry["command"], str):
                args = entry["command"].split()[1:]
            file_to_args[file_path] = args
            dir_to_args[os.path.dirname(file_path)] = args

        try:
            index = Index.create()
        except Exception:
            return {}

        bases: Dict[str, List[str]] = {}

        def best_args_for(path: str) -> List[str]:
            best_dir = ""
            best_args: List[str] = []
            for d, a in dir_to_args.items():
                if path.startswith(d) and len(d) > len(best_dir):
                    best_dir = d
                    best_args = a
            if not best_args and file_to_args:
                return next(iter(file_to_args.values()))
            return best_args

        # Parse the headers where classes are defined to get precise bases
        parsed = set()
        for (nm, rel), _ln in classes_sorted:
            header_path = os.path.normpath(os.path.join(doc_index, rel))
            if header_path in parsed:
                continue
            if "/csrc/" not in header_path:
                continue
            parsed.add(header_path)
            args = best_args_for(header_path)
            tu = None
            try:
                tu = index.parse(
                    header_path,
                    args=args,
                    options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
                )
            except Exception:
                continue
            if not tu:
                continue

            def visit(cursor):
                try:
                    for c in cursor.get_children():
                        k = c.kind
                        if (
                            k in (CursorKind.STRUCT_DECL, CursorKind.CLASS_DECL)
                            and c.is_definition()
                        ):
                            loc_file = ""
                            try:
                                loc_file = (
                                    c.location.file.name
                                    if c.location and c.location.file
                                    else ""
                                )
                            except Exception:
                                pass
                            try:
                                samefile = (
                                    os.path.samefile(loc_file, header_path)
                                    if loc_file
                                    else False
                                )
                            except Exception:
                                samefile = loc_file == header_path
                            if samefile:
                                cname = c.spelling
                                base_names: List[str] = []
                                for ch in c.get_children():
                                    if ch.kind == CursorKind.CXX_BASE_SPECIFIER:
                                        try:
                                            t = ch.type
                                            spelling = (
                                                t.spelling
                                                if (t is not None and t.spelling)
                                                else ""
                                            )
                                            if (
                                                not spelling
                                                and ch.referenced is not None
                                            ):
                                                spelling = ch.referenced.spelling or ""
                                            if spelling:
                                                base_names.append(
                                                    spelling.split("::")[-1]
                                                )
                                        except Exception:
                                            pass
                                if cname and cname not in bases:
                                    bases[cname] = base_names
                        visit(c)
                except Exception:
                    pass

            visit(tu.cursor)
        return bases

    base_map = collect_bases()

    # Supplement with a lightweight header parser to fill any bases that
    # libclang didn't resolve (clang is still required to be present).
    def collect_bases_fallback() -> Dict[str, List[str]]:
        bases_fb: Dict[str, List[str]] = {}
        token_strip = {
            "final",
            "NVF_API",
            "NVF_HOST_DEVICE",
            "NVF_API_T",
            "__declspec",
            "__attribute__",
        }

        def parse_bases(header: str) -> List[str]:
            header_nocom = re.sub(r"/\*.*?\*/", " ", header, flags=re.DOTALL)
            header_nocom = re.sub(r"//.*", " ", header_nocom)
            colon = header_nocom.find(":")
            if colon == -1:
                return []
            tail = header_nocom[colon + 1 :]
            brace = tail.find("{")
            if brace != -1:
                tail = tail[:brace]
            out: List[str] = []
            cur = []
            ang = 0
            par = 0
            for ch in tail:
                if ch == "<":
                    ang += 1
                elif ch == ">":
                    ang = max(0, ang - 1)
                elif ch == "(":
                    par += 1
                elif ch == ")":
                    par = max(0, par - 1)
                if ch == "," and ang == 0 and par == 0:
                    out.append("".join(cur))
                    cur = []
                else:
                    cur.append(ch)
            if cur:
                out.append("".join(cur))
            bases_clean: List[str] = []
            for item in out:
                s = " ".join(item.replace("\n", " ").split())
                for acc in ("public", "protected", "private", "virtual"):
                    s = s.replace(acc + " ", " ")
                for tok in token_strip:
                    s = s.replace(tok, "")
                s = s.strip()
                if not s:
                    continue
                bases_clean.append(s.split("::")[-1])
            return [b for b in bases_clean if b]

        for (name, rel), ln in classes_sorted:
            abs_path = os.path.normpath(os.path.join(doc_index, rel))
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue
            i = max(0, ln - 1)
            header_chunks: List[str] = []
            found = False
            j = i
            while j < len(lines) and j < i + 20:
                s = lines[j].rstrip("\n")
                header_chunks.append(s)
                if "{" in s:
                    found = True
                    break
                j += 1
            if not found:
                continue
            header = " ".join(header_chunks)
            if not re.search(
                r"\b(class|struct)\b[^\{]*\b" + re.escape(name) + r"\b", header
            ):
                continue
            bases_here = parse_bases(header)
            if bases_here:
                bases_fb.setdefault(name, bases_here)
        return bases_fb

    fb_map = collect_bases_fallback()
    for k, v in fb_map.items():
        if k not in base_map or not base_map[k]:
            base_map[k] = v

    def base_chain_for(name: str) -> str:
        # Single-inheritance chain pretty printer; multiple bases listed after ':'
        visited = set()
        chain = [name]
        current = name
        while current in base_map and len(base_map[current]) == 1:
            base = base_map[current][0]
            if base in visited:
                break
            visited.add(base)
            chain.append(base)
            current = base
        if len(chain) > 1:
            return " -> ".join(chain)
        bases_here = base_map.get(name, [])
        if bases_here:
            return f"{name} : {', '.join(bases_here)}"
        return ""

    # Kind-based grouping using heuristics from relative path
    def rel_segments(rel_path: str) -> List[str]:
        parts = rel_path.split("/")
        try:
            idx = parts.index("csrc")
            return parts[idx + 1 :]
        except ValueError:
            return parts

    def classify_kind(rel_path: str) -> str:
        segs = rel_segments(rel_path)
        filename = segs[-1] if segs else rel_path
        # Specific hierarchical directories
        if "ir" in segs:
            return "IR"
        if "kernel_ir" in segs or filename.startswith("kernel_ir"):
            return "Kernel IR"
        if "host_ir" in segs:
            return "Host IR"
        if "val_graph" in segs or filename.startswith("val_graph"):
            return "IR (Val Graph)"
        if "scheduler" in segs or "preseg_passes" in segs:
            return "Scheduler"
        if "serde" in segs:
            return "Serialization"
        if "ops" in segs:
            return "Ops"
        if (
            "runtime" in segs
            or filename.startswith("driver_api")
            or "multidevice" in segs
        ):
            return "Runtime"
        if (
            "device_lower" in segs
            or filename.startswith("codegen")
            or (
                filename.startswith("kernel")
                and filename
                not in (
                    "kernel_ir.h",
                    "kernel_ir.cpp",
                    "kernel_ir_dispatch.h",
                    "kernel_ir_dispatch.cpp",
                )
            )
            or "kernel_db" in segs
        ):
            return "Lowering/Codegen"
        analysis_hints = (
            "alias_analysis",
            "interval_analysis",
            "contiguity",
            "index_compute",
            "logical_domain_map",
            "parallel_dimension_map",
            "predicate_compute",
            "vectorization_info",
            "validator_utils",
            "expr_simplifier",
            "expr_evaluator",
            "id_model",
            "graph_traversal",
        )
        for hint in analysis_hints:
            if any(hint in s for s in segs) or filename.startswith(hint):
                return "Analysis"
        util_hints = (
            "utils",
            "options",
            "debug",
            "instrumentation",
            "macros",
            "visibility",
            "type",
            "type_promotion",
            "polymorphic_value",
            "linked_hash_map",
            "disjoint_set",
            "statement_guard",
            "bfs",
            "sys_utils",
            "opaque_type",
        )
        for hint in util_hints:
            if any(hint in s for s in segs) or filename.startswith(hint):
                return "Utilities/Core"
        return "Other"

    from collections import defaultdict

    classes_by_group: DefaultDict[str, List[Tuple[Tuple[str, str], int]]] = defaultdict(
        list
    )
    for item in classes_sorted:
        (name, rel), ln = item
        classes_by_group[classify_kind(rel)].append(item)

    enums_by_group: DefaultDict[str, List[Tuple[Tuple[str, str], int]]] = defaultdict(
        list
    )
    for item in enums_sorted:
        (name, rel), ln = item
        enums_by_group[classify_kind(rel)].append(item)

    # Build a name -> (rel, ln) map (first occurrence wins) for hierarchy
    name_to_item: Dict[str, Tuple[str, int]] = {}
    for (name, rel), ln in classes_sorted:
        name_to_item.setdefault(name, (rel, ln))

    # Choose a single immediate parent per class for tree rendering
    immediate_parent: Dict[str, str] = {}
    for name in name_to_item.keys():
        bases = base_map.get(name, [])
        parent = None
        for b in bases:
            if b in name_to_item:
                parent = b
                break
        if parent:
            immediate_parent[name] = parent

    # Build children mapping (include cross-kind) for recursive rendering
    children_by_name: DefaultDict[str, List[str]] = defaultdict(list)
    for child, parent in immediate_parent.items():
        children_by_name[parent].append(child)
    for v in children_by_name.values():
        v.sort(key=lambda x: x.lower())

    def render_tree(out, root_name: str, depth: int = 0) -> None:
        rel, ln = name_to_item[root_name]
        topic_rel = f"type_index/{safe_topic_filename(root_name)}"
        indent = "  " * depth
        out.write(f"{indent}- [{root_name}]({rel}#L{ln}) [(info)]({topic_rel})\n")
        for child in children_by_name.get(root_name, []):
            render_tree(out, child, depth + 1)

    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

    def safe_topic_filename(name: str) -> str:
        s = name.replace("/", "_slash_")
        s = s.replace(" ", "_")
        return s.lower() + ".md"

    order = [
        "IR",
        "Kernel IR",
        "Host IR",
        "IR (Val Graph)",
        "Scheduler",
        "Analysis",
        "Lowering/Codegen",
        "Ops",
        "Serialization",
        "Runtime",
        "Utilities/Core",
        "Other",
    ]

    os.makedirs(doc_index, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# API Topics\n\n")
        out.write(
            "Autogenerated index of types under `csrc` with links to their defining files.\n\n"
        )
        present_kinds = [g for g in order if g in classes_by_group]
        counts_by_kind = {
            g: len(classes_by_group.get(g, [])) + len(enums_by_group.get(g, []))
            for g in order
        }
        if present_kinds:
            out.write("## Table of contents (Kinds)\n\n")
            for g in present_kinds:
                out.write(f"- [{g} ({counts_by_kind.get(g, 0)})](#kind-{slug(g)})\n")
            out.write("\n")
        out.write("## Types (classes/structs)\n\n")
        for group in [g for g in order if g in classes_by_group]:
            out.write(f'<a id="kind-{slug(group)}"></a>\n')
            out.write(f"### {group} ({len(classes_by_group[group])})\n\n")
            # roots for this group are names classified to this group whose immediate parent
            # is either absent or classified to a different group
            group_names = {name for (name, _rel), _ln in classes_by_group[group]}
            roots = []
            for name in sorted(group_names, key=lambda x: x.lower()):
                parent = immediate_parent.get(name)
                if not parent:
                    roots.append(name)
                    continue
                # place under parent even if parent belongs to another group; don't render as root here
                parent_rel = (
                    name_to_item.get(parent, (None, None))[0]
                    if parent in name_to_item
                    else None
                )
                parent_group = classify_kind(parent_rel) if parent_rel else None
                if parent_group != group:
                    # parent not in this group; treat as root here
                    roots.append(name)
            for r in roots:
                render_tree(out, r, depth=0)
            out.write("\n")

        out.write("## Enums\n\n")
        for group in [g for g in order if g in enums_by_group]:
            out.write(f"### {group} ({len(enums_by_group[group])})\n\n")
            for (name, rel), ln in enums_by_group[group]:
                topic_rel = f"type_index/{safe_topic_filename(name)}"
                out.write(f"- [{name} (Enum)]({rel}#L{ln}) [(info)]({topic_rel})\n")
            out.write("\n")

    print(
        f"Wrote {len(classes_sorted)} types and {len(enums_sorted)} enums to {out_path}"
    )


if __name__ == "__main__":
    sys.exit(main())
