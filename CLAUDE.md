
## Memory (Mem-Engram)

You have persistent project memory via MCP tools. **These are mandatory behaviors, not optional.**

### At Session Start (ALWAYS do this first)
- Call `mem_context` to load previous work context BEFORE responding to the first user message.
- This gives you continuity from previous sessions (what was done, what's pending, warnings).

### During Session
- Call `mem_search` before starting tasks to check for relevant knowledge, procedures, or warnings.
- Call `mem_event(event_type, content)` to record important observations, decisions, and task outcomes.
- When you learn something new about the project, call `mem_store` to save it.

### After Completing or Failing a Task
- Call `mem_feedback(description, memories_used, status)` to report the outcome.
  - "success" → memories that helped get confidence boost
  - "failure" → memories that misled get corrected, failure case is created
  - "partial" → caveats are added
- This is how the memory system learns and improves. NEVER skip this step.

### When User Says "Remember..." or States a Preference
- Call `mem_instruct(rule, scope)` to save the rule/preference permanently.

### Rules
- ALWAYS call `mem_context` at session start — this is non-negotiable.
- ALWAYS call `mem_search` before deployments, debugging, or infrastructure changes.
- ALWAYS call `mem_feedback` after task completion — even for small tasks.
- When you encounter an error, `mem_search` for similar past failures before debugging from scratch.
