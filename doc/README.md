<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# nvFuser Documentation

## Purpose and Goals

This documentation is designed to provide high-level architectural context and design principles that are not easily discernible from reading individual source code files. It serves multiple critical purposes:

### AI-Assisted Development
- Provides rich context for AI agents to generate more accurate and coherent code
- Enables better code generation through enhanced context awareness
- Facilitates easier documentation updates by correlating code changes with architectural context
- Helps identify when code changes invalidate existing documentation

### Learning and Onboarding
- Serves as an interactive tutorial for new developers
- Enables AI agents to guide developers through the codebase effectively
- Helps developers quickly understand nvFuser's design principles
- Provides context for making design decisions that align with nvFuser's architecture

### API Development
- Supports the creation of high-quality user-facing APIs (e.g., Python bindings)
- Ensures new APIs maintain architectural consistency
- Provides context for API design decisions
- Helps maintain coherence between different API layers

## Documentation Structure

Table of Contents:

- Architecture Documentation ([architecture](architecture/))
  - [System Overview](architecture/system-overview.md)
  - [Design Principles](architecture/design-principles.md)
  - [Major Systems](architecture/fusion-system.md)
  - [C++ API](architecture/api-overview.md)
- Basic Notes to Developers ([dev](dev/))
  - [Symbol Visibility](dev/visibility.md)
  - [Introduction to TMA Support in NVFuser](dev/tma.md)
  - [Debugging](dev/debug.md)
- Deeper Reading Materials ([reading](reading/))
  - [Divisibility of Split](reading/divisibility-of-split.md)
  - [TMA Modeling In Depth](reading/tma-modeling-in-depth.md)
- Mathematical Background ([math](math/))
  - [Mathematical Logic](math/logic.md)
  - [Monotonic Function](math/monotonic-function.md)
  - [A Brief Overview of Abstract Algebra](math/abstract-algebra.md)
  - [Integer Division](math/integer-division.md)
- Appendix ([appendix](appendix/))
  - [AI Development Guidelines](appendix/ai-guidelines.md)
  - [Prompt Templates](appendix/standard-prompt-template.md)

## Using This Documentation

### For Developers
- Start with the Architecture Documentation for a high-level understanding
- Use the Basic Notes for specific development tasks
- Refer to Deeper Reading Materials for complex concepts
- Consult Mathematical Background for theoretical foundations

### For AI Agents
- Use the architecture context to generate more accurate code
- Maintain consistency with documented design principles
- Help identify documentation that needs updates
- Guide developers through the codebase using documented context

### For API Developers
- Ensure new APIs align with architectural principles
- Use documentation to maintain consistency across API layers
- Leverage context for better API design decisions
- Keep documentation in sync with API changes
