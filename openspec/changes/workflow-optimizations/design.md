## Context
Development workflows currently require running multiple commands manually (format, lint, type check, test) before committing. Common tasks like adding TODOs, checking project status, or running quick evaluations require multiple steps. Agent orchestration lacks explicit guidelines for quality gates, context management, and error recovery, leading to inefficient workflows and potential quality issues.

This proposal adds slash commands for common workflows and enhances CLAUDE.md with workflow patterns, quality checkpoints, and conventions.

## Goals / Non-Goals

**Goals:**
- Reduce friction in common development workflows through slash commands
- Prevent quality issues through explicit checkpoints
- Improve agent efficiency through context management hints
- Standardize git workflow conventions
- Provide error recovery playbooks for common failures
- Optimize agent orchestration with workflow patterns
- Guide cost/performance trade-offs with model selection guidance

**Non-Goals:**
- Create complex tooling or external dependencies (keep it simple)
- Replace existing tools (use slash commands to orchestrate existing tools)
- Automate everything (some manual verification is good)
- Create MCP servers (use simple bash in slash commands for now)

## Decisions

### Decision: Slash commands for common workflows
Create slash commands in `.claude/commands/` for frequently-used multi-step workflows.

**Rationale:**
- Single command vs multiple manual commands (faster, less error-prone)
- Project-specific workflows that don't fit general-purpose agents
- No external dependencies (just bash scripts)
- Easy to create and maintain

**Commands to create:**
- `/check` - All quality checks (most frequently needed)
- `/todo` - Quick task capture
- `/idea` - Quick improvement capture
- `/status` - Project overview
- `/sync` - Pull and update dependencies
- `/clean` - Clean artifacts

**Alternatives considered:**
- MCP servers: Rejected as too complex for simple workflows
- Custom scripts in `scripts/`: Rejected as less discoverable than slash commands
- Bash aliases: Rejected as not available to agents

### Decision: Interactive vs argument-based slash commands
Use interactive prompts for `/todo` and `/idea`, arguments for others.

**Rationale:**
- `/todo` and `/idea` benefit from structured prompts (priority, category, description)
- Other commands have fixed behavior (no parameters needed)
- Interactive is more user-friendly for infrequent use
- Can add argument support later if needed

**Implementation:**
```bash
# /todo interactive
read -p "Priority (high/medium/low): " priority
read -p "Task: " task
# Append to TODO.md under appropriate section
```

### Decision: Quality checkpoints in CLAUDE.md
Add explicit "Before X" checklists to CLAUDE.md for critical workflow points.

**Rationale:**
- Agents currently lack explicit quality gates
- Prevents broken commits, incomplete proposals, unfinished tasks
- Makes quality requirements explicit and auditable
- Agents can reference these checklists in their workflow

**Checkpoints:**
- Before committing (tests, linting, docs updated)
- Before creating OpenSpec proposal (check for duplicates, validate)
- Before marking task complete (all sub-tasks done, no commented code)

### Decision: Context management hints
Document which files should be read together for common tasks.

**Rationale:**
- Agents sometimes read files individually when they should read related files together
- Reduces back-and-forth and incomplete context
- Faster workflows with fewer file reads
- Serves as documentation of module relationships

**Implementation:**
- Group by task type (BPM work, external services, evaluation)
- List 3-5 related files per group
- Include brief reason why they're related

### Decision: Error recovery playbooks
Provide step-by-step recovery procedures for common failures.

**Rationale:**
- Agents currently improvise when errors occur
- Standard procedures are faster and more reliable
- Reduces debugging time
- Serves as troubleshooting guide for humans too

**Playbooks for:**
- Test failures (most common)
- Import/dependency errors (common after changes)
- OpenSpec validation failures (common during proposals)

### Decision: Agent workflow patterns
Document standard patterns for multi-agent orchestration.

**Rationale:**
- Current CLAUDE.md has good parallelization guidance but lacks workflow patterns
- Standardizes how agents hand off work
- Improves efficiency of complex multi-step tasks
- Makes expectations explicit

**Patterns:**
- Code review → implementation
- Analysis → proposal → implementation
- Parallel work for independent components

### Decision: Git workflow conventions
Document branch naming, commit granularity, and push timing.

**Rationale:**
- Currently not specified in CLAUDE.md
- Ensures consistent git history
- Prevents large monolithic commits
- Clarifies when work is "done enough" to push

**Conventions:**
- Branch naming: feature/fix/refactor/docs prefixes
- Commit granularity: One logical change per commit
- Push timing: After completed tasks, not with broken tests

### Decision: Agent model selection guidance
Document when to use Haiku vs Sonnet vs Opus.

**Rationale:**
- Current CLAUDE.md mentions preferring Haiku for speed but lacks specifics
- Helps optimize cost/performance trade-offs
- Makes agent spawning decisions explicit
- Prevents over-using expensive models

**Guidance:**
- Haiku: Exploration, simple tasks, file operations
- Sonnet: Implementation, reviews, proposals (default)
- Opus: Complex architecture, difficult debugging (rare)

## Structure

### Slash Commands

```
.claude/commands/
├── check.md          # All quality checks
├── todo.md           # Add to TODO.md
├── idea.md           # Add to IDEAS.md
├── status.md         # Project overview
├── sync.md           # Pull and sync deps
├── clean.md          # Clean artifacts
└── openspec/         # Existing OpenSpec commands
```

### CLAUDE.md Structure (Updated)

CLAUDE.md will have these sections after enhancement:

```markdown
<!-- OPENSPEC:START -->
[Existing OpenSpec managed block]
<!-- OPENSPEC:END -->

# Task Execution
[Existing content, enhanced model selection]

## Development Context
[Existing content]

# Interaction Style
[Existing content]

# Git Commits
[Existing content - KEEP AS-IS]

# Git Workflow (NEW)
## Branch Naming
## Commit Granularity
## When to Push

# Quality Checkpoints (NEW)
## Before Committing
## Before Creating OpenSpec Proposal
## Before Marking Task Complete

# Critical Files for Common Tasks (NEW)
## BPM-related Work
## External Services
## Evaluation Changes

# Error Recovery Playbooks (NEW)
## Test Failures
## Import/Dependency Errors
## OpenSpec Validation Failures

# Agent Workflow Patterns (NEW)
## Code Review → Implementation
## Analysis → Proposal → Implementation
## Parallel Work

# Task Management (NEW)
## OpenSpec for Features
## TODO.md for Small Tasks
## IDEAS.md for Improvements

# Detailed Guidelines
[Existing content]

# Commands Reference
[Existing content]
```

**Estimated new length:** ~200-250 lines (still manageable in single file)

## Risks / Trade-offs

**Risk:** Slash commands may need bash-specific features not portable
- **Mitigation:** Keep commands simple, use POSIX-compatible bash
- **Mitigation:** Test on target environment (WSL2/Linux)

**Risk:** Interactive slash commands may be annoying for frequent use
- **Mitigation:** Start with interactive, add argument support if needed
- **Mitigation:** Keep prompts minimal (2-3 inputs max)

**Risk:** Quality checkpoints may slow down workflow
- **Mitigation:** Checkpoints should be quick to verify (<30 seconds)
- **Mitigation:** Agents can run checks in parallel
- **Trade-off:** Slower workflow but fewer broken commits

**Risk:** Too many workflow patterns may be overwhelming
- **Mitigation:** Focus on 3-5 most common patterns only
- **Mitigation:** Keep patterns concise (3-5 steps each)

### Decision: Add workflow optimizations to CLAUDE.md
Add all workflow optimizations directly to CLAUDE.md rather than creating a separate workflow guide.

**Rationale:**
- CLAUDE.md is currently short (~71 lines)
- These are high-impact, frequently-referenced patterns
- Single file is easier to discover and maintain
- CLAUDE.md is already the authoritative agent instruction file
- Can extract to separate guide later if it grows too large (>500 lines)

**Structure in CLAUDE.md:**
- Existing sections stay (OpenSpec, Task Execution, Development Context, etc.)
- New sections added after existing content
- Clear section headers for easy navigation
- Keep each section concise (5-15 lines per section)

## Migration Plan

### Phase 1: Slash Commands
1. Create 7 slash command files
2. Test each command individually
3. Test interactive commands (todo, idea)

### Phase 2: CLAUDE.md Enhancements
4. Add quality checkpoints section
5. Add context management hints section
6. Add error recovery playbooks section
7. Add agent workflow patterns section
8. Add git workflow conventions section
9. Add agent model selection section

### Phase 3: Validation
10. Test slash commands in actual workflows
11. Verify agents reference checkpoints appropriately
12. Test error recovery playbooks with actual errors
13. Verify workflow patterns improve orchestration

### Rollback
- Slash commands: Just delete `.claude/commands/*.md` files
- CLAUDE.md additions: Revert to previous version

## Open Questions
None - approach is clear and implementation is straightforward.
