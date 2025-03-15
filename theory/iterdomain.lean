import Mathlib.Algebra.Order.Floor.Div
import Mathlib.Data.Vector.Basic
import Mathlib.Tactic.Basic

structure IterDomainTransformations (n m : Nat) where
  extent_map : Vector Nat n -> Vector Nat m
  index_map : Vector Nat n -> Vector Nat m

def equiv {n m : Nat} (t1 t2 : IterDomainTransformations n m) : Prop :=
  t1.extent_map = t2.extent_map ∧ t1.index_map = t2.index_map

def equiv_refl {n m : Nat} (t : IterDomainTransformations n m) : equiv t t :=
  And.intro rfl rfl

def equiv_symm {n m : Nat} (t1 t2 : IterDomainTransformations n m) : equiv t1 t2 → equiv t2 t1 :=
  sorry

def equiv_trans {n m : Nat} (t1 t2 t3 : IterDomainTransformations n m) : equiv t1 t2 → equiv t2 t3 → equiv t1 t3 :=
  sorry

def inner_split (factor : Nat) : IterDomainTransformations 1 2 :=
  { extent_map := fun v => Vector.cons (v[0] ⌈/⌉ factor) (Vector.cons factor Vector.nil),
    index_map := fun v => Vector.cons (v[0] / factor) (Vector.cons (v[0] % factor) Vector.nil) }

def _slice {α : Type} {n : Nat} (left right : Nat) (v : Vector α (left + n + right)) : Vector α n :=
  have h : left + n + right - left = n + right := by
    rw [Nat.sub_add_comm (Nat.le_add_right left n)]
    rw [Nat.add_sub_cancel_left]
  cast (by simp [h]) ((v.drop left).take n)

def extend {n m : Nat} (t : IterDomainTransformations n m) (left right : Nat) : IterDomainTransformations (left + n + right) (left + m + right) :=
  let apply_slice := fun (f : Vector Nat n -> Vector Nat m) =>
    fun (v : Vector Nat (left + n + right)) =>
      have h: 0 + left + (n + right) = left + n + right := by
        rw [Nat.zero_add]
        rw [Nat.add_assoc]
      let prefix_ : Vector Nat left := _slice 0 (n + right) (cast (by rw [h]) v)
      let suffix : Vector Nat right := _slice (left + n) 0 v
      let middle : Vector Nat m := f (_slice left right v)
      (prefix_.append middle).append suffix
  { extent_map := apply_slice t.extent_map,
    index_map := apply_slice t.index_map }

def compose {n m l : Nat} (t1 : IterDomainTransformations m l) (t2 : IterDomainTransformations n m) : IterDomainTransformations n l :=
  { extent_map := fun v => t1.extent_map (t2.extent_map v),
    index_map := fun v => t1.index_map (t2.index_map v) }

theorem split_split {n m : Nat}:
  let t1 := compose (extend (inner_split m) 0 1) (inner_split n)
  let t2 := compose (extend (inner_split n) 1 0) (inner_split (m * n))
  equiv t1 t2
:=
  let t1 := compose (extend (inner_split m) 0 1) (inner_split n)
  let t2 := compose (extend (inner_split n) 1 0) (inner_split (m * n))
  have hem : t1.extent_map = t2.extent_map :=
    have houter : ∀ i : Nat, i ⌈/⌉ m ⌈/⌉ n = i ⌈/⌉ (m * n) := sorry
    sorry
  have him : t1.index_map = t2.index_map :=
    have h1 : ∀ i : Nat, i / n / m = i / (m * n) := sorry
    have h2 : ∀ i : Nat, i / n % m = i % (m * n) / n := sorry
    have h3 : ∀ i : Nat, i % n = i % (m * n) % n := sorry
    sorry
  And.intro hem him

-------------------------------------

variables (f1 g1 f2 g2 : ℕ → ℕ)

def ff (f1 f2 : ℕ → ℕ) : Vector ℕ 1 → Vector ℕ 2 :=
  fun x => ⟨[f1 (x.head), f2 (x.head)], by simp⟩

def gg (g1 g2 : ℕ → ℕ) : Vector ℕ 1 → Vector ℕ 2 :=
  fun x => ⟨[g1 (x.head), g2 (x.head)], by simp⟩

theorem tt (h : f1 = g1 ∧ f2 = g2) : ff f1 f2 = gg g1 g2 :=
  funext (fun x =>
    match x with
    | ⟨[a], _⟩ => by
        simp [ff, gg, h.1, h.2])
