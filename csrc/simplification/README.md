# E-Graph Expression Simplification

This directory contains an implementation of E-Graphs[^1] that is customized
for simplifying `Val*` objects in nvFuser.

In this directory:
- `egraph.h`: Contains the main class `EGraphSimplifier`. Start here.
- `enode.h`: Defines the `ASTNode` and `ENode` classes that determine how our AST is modelled.
- `eclass.h`: Defines the `EClass` class, and basic generic EClass functionality. This does not contain many nvFuser-specific modifications compared to [^1].
- `rules.cpp`


[^1]: [Willsey et al. egg: Fast and Extensible Equality Saturation. POPL 2021.](https://doi.org/10.5281/zenodo.4072013)
