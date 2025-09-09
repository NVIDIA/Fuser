# NVFuser Workshop Docs Index

Start here:
- [TOPICS.md](./TOPICS.md)

## Getting Started and Environment
- [Getting started: tv_add_2d](./getting-started-tv-add-2d.md)
- [Build and run standalone samples](./build-and-run-standalone-samples.md)
- [Runtime env issues and IR-only mode](./runtime-env-and-ir-only-mode.md)

## Fusion IR and Reading Dumps
- [Fusion IR anatomy and printer semantics](./fusion-ir-anatomy-and-printer.md)
- [How to read Fusion IR dumps](./how-to-read-fusion-ir-dumps.md)
- [Dump and inspect IR (prints and getters)](./dump-and-inspect-ir.md)

## View Ops and Shape Mechanics
- [What is a ViewAnalysis, and how it’s used](./what-is-view-analysis.md)
- [Empty dimensions in reshape](./reshape-empty-dimensions.md)
- [Squeeze (and Broadcast)](./squeeze-and-broadcast.md)

## TensorViews, IR Building, and Execution
- [TensorView mental model (symbolic vs data)](./tensorview-mental-model.md)
- [Instantiate TVs, bind tensors, execute via FEC](./instantiate-bind-execute.md)
- [Build IR with FusionGuard and free functions](./build-ir-with-fusion-guard.md)

## Scheduling, Vectorization, Correctness
- [Split divisibility and predication](./split-divisibility-and-predication.md)
- [Vectorization alignment and tails](./vectorization-alignment-and-tails.md)
- [Weak vs strong correctness](./weak-vs-strong-correctness.md)

## TMA and Tensor Core Transfers
- [TMA FTTC and weak correctness](./tma-fttc-and-weak-correctness.md)
- [TMA consumer schedule constraints](./tma-consumer-schedule-constraints.md)
- [LdMatrix/StMatrix inner-loop parallelization](./ldmatrix-stmatrix-parallelization.md)
