import Mathlib.Data.Real.Basic

def strictly_increasing(f: ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def weakly_increasing(f: ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem t1 {f : ℝ → ℝ}:
  strictly_increasing f →
  (∀ x y : ℝ, x < y ↔ f x < f y) ∧
  (∀ x y : ℝ, x ≤ y ↔ f x ≤ f y)
:=
  fun hsi : strictly_increasing f =>
  have hle : ∀ x y : ℝ, x ≤ y → f x ≤ f y :=
    fun hx hy : ℝ =>
    have hltim : hx < hy -> f hx ≤ f hy :=
      fun hxhy : hx < hy => le_of_lt ((hsi hx hy) hxhy)
    have heqim : hx = hy -> f hx ≤ f hy :=
      fun hxhy : hx = hy => le_of_eq (congrArg f hxhy)
    fun hle : hx ≤ hy => hle.lt_or_eq_dec.elim hltim heqim
  have hrev : ∀ x y : ℝ, f x < f y → x < y :=
    fun x y : ℝ =>
    fun hflt : f x < f y =>
    have nle : ¬ y ≤ x:=
      fun hrevle : y ≤ x =>
      absurd ((hle y x) hrevle) hflt.not_le
    lt_of_not_ge nle
  have hlerev : ∀ x y : ℝ, f x ≤ f y → x ≤ y :=
    fun x y : ℝ =>
    fun hflt : f x ≤ f y =>
    have nlt : ¬ y < x:=
      fun hrevlt : y < x =>
      absurd ((hsi y x) hrevlt) hflt.not_lt
    le_of_not_gt nlt
  have h1 : ∀ x y : ℝ, x < y ↔ f x < f y :=
    fun x y : ℝ => Iff.intro (hsi x y) (hrev x y)
  have h2 : ∀ x y : ℝ, x ≤ y ↔ f x ≤ f y :=
    fun x y : ℝ => Iff.intro (hle x y) (hlerev x y)
  And.intro h1 h2
