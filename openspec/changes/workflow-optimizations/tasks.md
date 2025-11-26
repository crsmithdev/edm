## 1. Slash Command Creation
- [x] 1.1 Create `.claude/commands/check.md` - run all quality checks (format, lint, type check, tests)
- [x] 1.2 Create `.claude/commands/todo.md` - add item to TODO.md interactively
- [x] 1.3 Create `.claude/commands/idea.md` - add improvement idea to IDEAS.md
- [x] 1.4 Create `.claude/commands/status.md` - comprehensive project status (git, openspec, todos, ideas)
- [x] 1.5 Create `.claude/commands/sync.md` - pull latest and sync dependencies
- [x] 1.6 Create `.claude/commands/clean.md` - clean build artifacts and caches

## 2. CLAUDE.md Enhancements
- [x] 2.1 Add "Git Workflow" section with branch naming, commit granularity, when to push
- [x] 2.2 Add "Quality Checkpoints" section with before committing, before proposal, before task complete
- [x] 2.3 Add "Critical Files for Common Tasks" section with BPM, external services, evaluation file groups
- [x] 2.4 Add "Error Recovery Playbooks" section with test failures, import errors, OpenSpec validation
- [x] 2.5 Add "Agent Workflow Patterns" section with code review→implementation, analysis→proposal→implementation, parallel work
- [x] 2.6 Add "Task Management" section with OpenSpec vs TODO.md vs IDEAS.md guidance
- [x] 2.7 Enhance existing "Task Execution" section with expanded model selection guidance (Haiku/Sonnet/Opus specifics)

## 3. Validation
- [x] 3.1 Test all slash commands execute correctly
- [x] 3.2 Verify slash commands handle errors gracefully
- [x] 3.3 Test interactive slash commands (todo, idea) work as expected
- [x] 3.4 Verify CLAUDE.md additions are clear and actionable
- [x] 3.5 Verify CLAUDE.md length is reasonable (< 300 lines)
- [x] 3.6 Test agent follows quality checkpoints when committing
