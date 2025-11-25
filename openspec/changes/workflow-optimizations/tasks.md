## 1. Slash Command Creation
- [ ] 1.1 Create `.claude/commands/check.md` - run all quality checks (format, lint, type check, tests)
- [ ] 1.2 Create `.claude/commands/todo.md` - add item to TODO.md interactively
- [ ] 1.3 Create `.claude/commands/idea.md` - add improvement idea to IDEAS.md
- [ ] 1.4 Create `.claude/commands/status.md` - comprehensive project status (git, openspec, todos, ideas)
- [ ] 1.5 Create `.claude/commands/sync.md` - pull latest and sync dependencies
- [ ] 1.6 Create `.claude/commands/clean.md` - clean build artifacts and caches

## 2. CLAUDE.md Enhancements
- [ ] 2.1 Add "Git Workflow" section with branch naming, commit granularity, when to push
- [ ] 2.2 Add "Quality Checkpoints" section with before committing, before proposal, before task complete
- [ ] 2.3 Add "Critical Files for Common Tasks" section with BPM, external services, evaluation file groups
- [ ] 2.4 Add "Error Recovery Playbooks" section with test failures, import errors, OpenSpec validation
- [ ] 2.5 Add "Agent Workflow Patterns" section with code review→implementation, analysis→proposal→implementation, parallel work
- [ ] 2.6 Add "Task Management" section with OpenSpec vs TODO.md vs IDEAS.md guidance
- [ ] 2.7 Enhance existing "Task Execution" section with expanded model selection guidance (Haiku/Sonnet/Opus specifics)

## 3. Validation
- [ ] 3.1 Test all slash commands execute correctly
- [ ] 3.2 Verify slash commands handle errors gracefully
- [ ] 3.3 Test interactive slash commands (todo, idea) work as expected
- [ ] 3.4 Verify CLAUDE.md additions are clear and actionable
- [ ] 3.5 Verify CLAUDE.md length is reasonable (< 300 lines)
- [ ] 3.6 Test agent follows quality checkpoints when committing
