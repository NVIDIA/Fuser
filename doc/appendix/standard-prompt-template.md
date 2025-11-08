# Standard Prompt Template for nvFuser Documentation

Use the following template when requesting the AI agent to generate or modify nvFuser documentation to ensure consistency and adherence to the practices outlined in ai-guidelines.md.

STANDARD PROMPT FOR NVFUSER DOCUMENTATION
----------------------------------------------------------------------

CONTEXT FOR AI:
This prompt is intended to be used while working on nvFuser documentation. The documentation helps users and developers understand nvFuser's architecture, design principles, and implementation details.

AI GUIDANCE CHECK:
Follow instructions in ai-guidelines.md. Key reminders:
  Clarity: Ask if the request is ambiguous.
  Placeholders: Use [[[TODO: Detail needed]]] to indicate where more information would be helpful.
  Edits (Propose for Review -> Apply Cleanly):
    Simple inline: Use ~~delete~~/**add** markup for review.
    Complex/multi-line: Apply simple changes with markup first. Then implement complex changes directly & add **AI PROPOSAL:** [Summary of complex changes] annotation.
    After confirmation: Apply cleanly (remove all markup and **AI PROPOSAL:** annotations).
  Next Step: Suggest a relevant next action after completing a task.

DEFAULTS (Override below if needed):
  Audience: nvFuser users and developers
  Tone: Clear, technical, precise, professional
  Documentation Structure: Follow the Documentation Structure Template in ai-guidelines.md

----------------------------------------------------------------------
USER REQUEST -- PLEASE FILL IN BELOW
----------------------------------------------------------------------

GOAL:
[Clearly state the primary objective (e.g., document component, update section, add examples)]

TARGET (Required):
[Specify target file path(s) and relevant section/location within the file(s)]

TASK DETAILS (Relevant to Goal):
[If documenting new content: Define component/topic and list key points/concepts/rules.]
[If updating existing content: Describe the desired change, e.g., "Improve clarity of section X," "Add performance implications to section Y."]
[If other task: Clearly describe the requested action.]

OPTIONAL OVERRIDES / SPECIFIC REQUESTS:
  Override Audience: [Specify if different from default]
  Override Tone: [Specify if different from default]
  Specific User Next Step Request: [Use this to request a specific action AFTER the main Goal is completed, overriding the AI's default proposal. E.g., "Perform a Technical Accuracy pass," "Stop and wait for next input."]

----------------------------------------------------------------------
