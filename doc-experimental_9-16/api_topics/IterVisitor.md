# IterVisitor

Source: [IterVisitor](../../csrc/iter_visitor.h#L39)

## Synopsis
- **Kind**: class (inherits from `OptOutDispatch`)
- **File**: `csrc/iter_visitor.h`
- **What it does**: Topological IR traversal from outputs toward inputs (DAG walk), invoking `handle(...)` in dependency order.

## Purpose
- Foundation for many IR analyses and transformations: sorting, dependency checks, inputs discovery, scheduling analyses.
- Provides traversal helpers: `traverse`, `traverseAllPaths`, `traverseTo`, `traverseBetween`; controls around traversing member vals, attributes, and multi-output siblings.

## Key concepts
- Traversal stack `stmt_stack` captures the active path and siblings during dispatch.
- Overrides `dispatch(Statement|Expr|Val)` to ensure outputs→inputs order.
- Pairs with `BackwardVisitor` for reverse-direction passes.

## Notable derivatives
- `StmtSort`, `InputsOf`, `DeadCodeRemover`, and numerous analysis passes (e.g., `ConcretizedBroadcastDomains`, `NonDivisibleSplitInfo`).

## See also
- [BackwardVisitor](../../csrc/iter_visitor.h#L182)
- [Statement](../../csrc/ir/base_nodes.h#L96), [Val](../../csrc/ir/base_nodes.h#L224), [Expr](../../csrc/ir/base_nodes.h#L505)
