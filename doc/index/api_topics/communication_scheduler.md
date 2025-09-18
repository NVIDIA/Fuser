# CommunicationScheduler

## Synopsis
Scheduler placeholder for communication-oriented graphs; integrates with registry and planning but may delegate minimal scheduling.

## Source
- Class: [`CommunicationScheduler`](../../../csrc/scheduler/communication.h#L26)

## Overview
Provides hooks for communication-heavy segments, enabling specialized parameterization and future extensions. Interfaces follow standard scheduler contract.

Interfaces:
- `canScheduleCompileTime/RunTime`
- `computeHeuristics(...)`
- `schedule(...)`
