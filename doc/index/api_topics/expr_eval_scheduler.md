# ExprEvalScheduler

## Synopsis
Special-case scheduler that evaluates expressions via the ExpressionEvaluator without generating a CUDA kernel.

## Source
- Class: [`ExprEvalScheduler`](../../../csrc/scheduler/expr_eval_sched.h#L21)

## Overview
Accepts specific op patterns (notably MatmulOp per header note) and materializes outputs using host-side evaluation. Used to short-circuit codegen when appropriate.

Interfaces:
- `canScheduleCompileTime/RunTime`
- `computeHeuristics(...)`
- `schedule(...)`
