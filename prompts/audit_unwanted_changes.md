# Audit Prompt: Find Unwanted Code Changes

Paste this into Claude after a coding session where you suspect unauthorized modifications.

---

## The Prompt

```
You are auditing the SETAI codebase for unauthorized or unwanted changes. Your job is to find anything that was added, removed, or modified WITHOUT my explicit request. Be paranoid and thorough.

Run `git diff` and `git diff --cached` to see all uncommitted changes. Then run `git log --oneline -20` to see recent commits.

For EVERY change you find, classify it as one of:

1. REQUESTED — I explicitly asked for this change
2. SIDE EFFECT — A necessary consequence of a requested change (e.g., fixing an import after moving a function)
3. SUSPICIOUS — Changed without clear justification. Flag these.
4. UNAUTHORIZED — Definitely not requested. These are the problems.

Specifically look for:

**Behavioral changes:**
- Modified thresholds, constants, or default values (epsilons, learning rates, loss weights, risk limits)
- Changed logic in if/else branches, especially in risk_manager.py, decision_fusion.py, brain_policy.py
- Added or removed function parameters that change call signatures
- Modified return values or their types
- Changed order of operations in the forward pass or decision pipeline

**Silent removals:**
- Deleted safety checks, guards, or validation logic
- Removed logging, warnings, or error messages
- Commented out code that was previously active
- Removed test cases or weakened assertions

**Scope creep:**
- "Refactored" or "cleaned up" files I didn't mention
- Added new files, classes, or functions I didn't ask for
- Changed formatting, imports, or style in files unrelated to the task
- Added dependencies to pyproject.toml I didn't request

**Dangerous patterns:**
- Hardcoded values replacing configurable ones (or vice versa)
- Changed exception handling (broad except where specific existed, or swallowed exceptions)
- Modified tensor operations that could affect numerical stability
- Altered async/threading behavior

For each SUSPICIOUS or UNAUTHORIZED change, show me:
- The exact file and line numbers
- What was there before vs what's there now
- Why you think it was changed (your best guess)
- Whether reverting it would break anything

Do NOT fix anything. Do NOT suggest improvements. Just report what you find.
```

---

## Shorter Version (Quick Audit)

```
Run git diff on the SETAI codebase. For every change, tell me: (1) which file, (2) what changed, (3) whether I explicitly asked for it. Flag anything I didn't request as UNAUTHORIZED. Pay special attention to: modified thresholds/constants, removed safety checks, changed logic branches, new files I didn't ask for, and "cleanup" in files unrelated to the task. Report only. Do not fix anything.
```

---

## Preventive Version (Give BEFORE a Task)

```
STRICT SCOPE RULES for this task:

1. ONLY modify the files I explicitly name
2. Do NOT refactor, rename, or "clean up" anything I didn't ask about
3. Do NOT change constants, thresholds, or default values unless I specifically say to
4. Do NOT add new dependencies, files, or classes unless I ask for them
5. Do NOT modify imports, formatting, or style in untouched code
6. If a change requires touching another file (e.g., updating an import), tell me first and wait for approval
7. If you see a bug or improvement opportunity outside my request, MENTION it but do NOT fix it
8. After completing the task, list every file you modified and summarize what you changed in each

Violating these rules is worse than not completing the task.
```
