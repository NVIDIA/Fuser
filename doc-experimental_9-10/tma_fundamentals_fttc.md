## TMA Fundamentals; FTTC and Efficient Weak Correctness

FTTC (impossibility of strong correctness) holds iff:
1) element stride e does not divide the tile box size B and e < B (e ∤ B and e < B)
2) tile box size B is smaller than tensor size S on that dim (B < S)

Implication: hardware zero‑fill cannot be applied precisely; accept weak correctness or predicate non‑TMA paths.

Efficient weak correctness: use TMA unpredicated; predicate only non‑TMA‑protected IDs to minimize overhead.

Refs: Q&A Topics 4–5 `../doc-bot/experimenting/8-28-2025/q_and_a.md`.


