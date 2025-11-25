## 1. Topic Documentation Creation
- [ ] 1.1 Create `docs/architecture.md` with system design, module organization, data flow, and tech stack rationale
- [ ] 1.2 Create `docs/cli-reference.md` with complete command documentation for analyze and evaluate subcommands
- [ ] 1.3 Create `docs/development.md` with testing workflows, logging patterns, code quality tools, and evaluation framework

## 2. Navigation Layer
- [ ] 2.1 Create `docs/agent-guide.md` as navigation/index for AI agents
- [ ] 2.2 Add "Quick Reference by Task" section with links to topic docs and brief context
- [ ] 2.3 Add "Documentation Map" table showing which doc to read when
- [ ] 2.4 Add file paths for key modules organized by feature
- [ ] 2.5 Keep agent-guide.md focused on navigation (no content duplication)

## 3. README Streamlining
- [ ] 3.1 Streamline README.md to: brief description (2-3 sentences), system requirements, installation steps, quick start
- [ ] 3.2 Add links to detailed documentation (cli-reference.md, architecture.md, development.md)
- [ ] 3.3 Keep most basic run/test commands in README with note to see cli-reference.md for complete docs
- [ ] 3.4 Move explanatory content (tool rationale, design decisions) to topic-specific docs

## 4. Task Tracking Files
- [ ] 4.1 Create `TODO.md` in repo root with structure: High/Medium/Low priority sections
- [ ] 4.2 Create `IDEAS.md` in repo root with structure: Ready for Proposal / Under Consideration / Icebox
- [ ] 4.3 Add initial examples to both files to demonstrate format

## 5. Integration
- [ ] 5.1 Add link to `docs/agent-guide.md` in CLAUDE.md as primary navigation for agents
- [ ] 5.2 Update CLAUDE.md to reference topic docs where appropriate
- [ ] 5.3 Add task management guidance to CLAUDE.md (OpenSpec for features, TODO.md for small tasks, IDEAS.md for improvements)

## 6. Validation
- [ ] 6.1 Verify all file paths referenced in documentation exist
- [ ] 6.2 Verify all CLI commands match current implementation
- [ ] 6.3 Verify code examples are syntactically valid
- [ ] 6.4 Verify no content duplication across documentation files
- [ ] 6.5 Test agent scenarios:
  - [ ] Agent can run analysis using only documentation
  - [ ] Agent can interpret evaluation results using only documentation
  - [ ] Agent can locate BPM detection code from functional description
  - [ ] Agent can run tests and understand workflows using only documentation
  - [ ] Agent can add items to TODO.md and IDEAS.md appropriately
