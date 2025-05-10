# Guidance for AI Agent Assisting with nvFuser Documentation

## Purpose

This document outlines specific guidance and best practices for AI agents assisting in the development and maintenance of nvFuser documentation. The goal is to ensure the AI's contributions are consistent, technically accurate, and maintain the architectural context that makes nvFuser's design principles clear to both human readers and other AI agents.

## AI Agent Best Practices / Instructions

* **Ask for clarification if the user's request is ambiguous.** Before proceeding with a request that seems unclear or lacks necessary detail, ask the user for clarification to ensure the output aligns with their intent.

* **Use explicit placeholders (`[[[TODO: ...]]]`) instead of inventing information.** When specific context or details are missing or unknown to the AI (e.g., specific implementation details, performance characteristics, or architectural decisions), insert a `[[[TODO: Specific detail needed]]]` marker instead of generating placeholder or potentially incorrect information.

* **Prefer clear, technical language.** Use precise technical terminology while maintaining readability. Balance high-level architectural concepts with implementation details.

* **Clearly propose textual changes for review, then apply cleanly after confirmation:**
  * **Simple, inline changes (within a single paragraph):** Use Markdown strikethrough (`~~deleted text~~`) and bold (`**added text**`) markup for each individual change.
  * **Complex changes (multi-line, lists, code blocks, sections):** For complex changes, first apply any accompanying simple inline changes using the markup above. Then, implement the complex changes directly and add an `**AI PROPOSAL:** [Summary of complex change(s) made]` annotation.
  * **After user confirmation:** Apply the approved changes cleanly: remove all `~~strikethrough~~`/`**bold**` markup and remove any `**AI PROPOSAL:**` annotations.

* **Propose a logical next step.** After successfully completing a request, suggest a relevant next action based on the context (e.g., "Should we document the interaction with the scheduler next?", "Would you like to expand on the performance implications?", "Is there anything else needed for this component?").

## Standard Editorial Pipeline (User-Initiated)

After substantial new content has been generated, the following passes can be performed **sequentially, only upon explicit user request for each pass:**

1. **Technical Accuracy Pass:**
   * **Focus:** Review for technical accuracy, architectural consistency, and completeness
   * **Action:** Verify:
     - Correctness of technical concepts
     - Consistency with architectural principles
     - Completeness of explanations
     - Accuracy of code examples

2. **Context Integration Pass:**
   * **Focus:** Ensure proper integration with existing documentation
   * **Action:** Check for:
     - Proper cross-referencing
     - Consistent terminology
     - Logical flow between sections
     - Clear connection to architectural principles

3. **Clarity and Readability Pass:**
   * **Focus:** Improve clarity and accessibility
   * **Action:** Review for:
     - Clear explanations of complex concepts
     - Appropriate level of detail
     - Consistent formatting
     - Effective use of examples

## Documentation Structure Template

When documenting a component or system, follow this structure:

```markdown
# [Component Name] Documentation

## Goals and Motivation
- Why this component exists
- What problems it solves
- Key design decisions and trade-offs

## Core Principles
- Fundamental concepts
- Design constraints
- Key invariants

## Key Performance Tradeoffs
- Performance considerations
- Memory usage
- Computation vs. memory tradeoffs
- Scalability implications

## High-Level Workflow
1. Step-by-step overview
2. Key interactions
3. Data flow
4. Control flow

## Detailed Explanations
- In-depth technical details
- Implementation specifics
- Edge cases and considerations
- Performance implications

## Future Optimizations / TODOs
- Known limitations
- Potential improvements
- Open questions
- Future work
```

## Example Documentation

For examples of well-structured documentation, refer to:
- [Fusion Segmenter Notes](../architecture/fusion-segmenter.md)
- [Polymorphic Value Notes](../architecture/polymorphic-value.md)

These examples demonstrate:
- Clear organization of technical content
- Balance of high-level concepts and implementation details
- Effective use of examples and explanations
- Proper handling of performance considerations

## Contributing
When adding new documentation:
1. Follow the established markdown format
2. Include code examples where relevant
3. Link to related documentation
4. Update this README.md with any new sections
5. Maintain architectural context
6. Consider both human readers and AI agents
7. Document design decisions and trade-offs 

## Standard Prompt Template

For the standard template to use when requesting AI assistance with nvFuser documentation, please refer to [Standard Prompt Template](standard-prompt-template.md). This template ensures that:

1. All documentation requests follow a consistent format
2. The AI agent has clear context about the task
3. The documentation maintains nvFuser's technical standards
4. Changes are properly reviewed before being applied
5. The documentation structure remains consistent across all components 