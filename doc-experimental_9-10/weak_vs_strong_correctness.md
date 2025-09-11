## Weak vs Strong Correctness; Hole Semantics; Neutral Values

Definitions:
- Weak correctness: hole elements may contain arbitrary data; safe if never consumed unsafely
- Strong correctness: holes contain the correct neutral value for the op (0 for sum, 1 for prod, ±inf for min/max)

Why it matters:
- Unpredicated reductions require strong correctness when holes exist
- TMA/vectorization choices influence whether you must ensure strong correctness or predicate

Refs: Q&A Topic 5 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.


