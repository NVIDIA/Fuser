theorem t1 {p q r : Prop}: (p → (q ↔ r)) → (p ∧ q ↔ p ∧ r) :=
  fun hl : (p → (q ↔ r)) =>
  have pq2pr: p ∧ q → p ∧ r :=
    have pq2r: p ∧ q → r :=
      fun hpq: p ∧ q => (hl hpq.left).mp hpq.right
    fun hpq : p ∧ q => And.intro hpq.left (pq2r hpq)
  have pr2pq: p ∧ r → p ∧ q :=
    have pr2q: p ∧ r → q :=
      fun hpr: p ∧ r => (hl hpr.left).mpr hpr.right
    fun hpr : p ∧ r => And.intro hpr.left (pr2q hpr)
  show p ∧ q ↔ p ∧ r from Iff.intro pq2pr pr2pq
