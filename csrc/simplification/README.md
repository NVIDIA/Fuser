# E-Graph Expression Simplification

This directory contains an implementation of E-Graphs[^1] that is customized
for simplifying `Val*` objects in nvFuser.

In this directory:
- `egraph.h`: Contains the main class `EGraph`. Start here.
- `enode.h`: Defines the `ASTNode` and `ENode` classes that determine how our
  AST is modelled.
- `eclass.h`: Defines the `EClass` class, and basic generic EClass
  functionality. This does not contain many nvFuser-specific modifications
  compared to [^1].
- `rules.cpp`


[^1]: [Willsey et al. egg: Fast and Extensible Equality Saturation. POPL 2021.](https://doi.org/10.5281/zenodo.4072013)

## Ownership

All classes defined in this directory other than `EGraph` and `EGraphGuard` are
indirectly owned by `EGraph`. The following diagram illustrates the ownership tree:
```
EGraph
   ├── UnionFind<Id>
   ├── HashCons a.k.a. std::unordered_map<ENode, Id>
   ├── RuleRunner
   │      └── Rule
   ├── EClass
   │      └── AnalysisData
   └── ENode
          ├── FunctionDesc
          └── ASTNode
                 └── FunctionDesc
```

Note that this describes _ownership_, but these objects do refer to one
another. For example, AnalysisData contains a pointer to the best
representative ASTNode in the class.
