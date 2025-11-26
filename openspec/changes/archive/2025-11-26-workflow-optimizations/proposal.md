# Change: Workflow Optimizations for Development and Agent Orchestration

## Why
Development workflows currently require multiple manual commands for common tasks (test+lint+format, project status checks, task capture). Agent orchestration lacks explicit quality checkpoints, context management hints, and error recovery patterns, leading to inefficient workflows and potential quality issues. Git workflow conventions are not documented, leading to inconsistent commit and branch practices.

## What Changes
- Add slash commands for common development workflows (`/check`, `/todo`, `/idea`, `/status`, `/sync`, `/clean`)
- Enhance CLAUDE.md with workflow optimizations:
  - Quality checkpoints to prevent broken commits and incomplete work
  - Context management hints to guide agents on which files to read together
  - Error recovery playbooks for common failure scenarios
  - Agent workflow patterns for orchestration and handoffs
  - Git workflow conventions (branch naming, commit granularity, when to push)
  - Enhanced agent model selection guidance (expand existing section)
  - Task management guidance (OpenSpec vs TODO.md vs IDEAS.md)

## Impact
- Affected specs: `development-workflow` (new capability)
- Affected code: None (documentation and tooling only)
- Affected files:
  - New: `.claude/commands/check.md`, `.claude/commands/todo.md`, `.claude/commands/idea.md`, `.claude/commands/status.md`, `.claude/commands/sync.md`, `.claude/commands/clean.md`
  - Modified: `CLAUDE.md` (add 7 new sections: quality checkpoints, context hints, error recovery, workflow patterns, git conventions, task management; enhance existing model selection section)
