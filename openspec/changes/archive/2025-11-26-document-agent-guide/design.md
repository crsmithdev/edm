## Context
Documentation is currently fragmented across README.md, CLAUDE.md, docs/*, and openspec/*, making it inefficient for both AI agents and humans to find information. Agents need comprehensive architecture and workflow documentation. Humans need quick reference without redundant explanations.

This proposal creates structured documentation with single sources of truth for each topic, eliminating duplication and fragmentation.

## Goals / Non-Goals

**Goals:**
- Create navigation guide for AI agents (agent-guide.md) that routes to appropriate detailed docs
- Establish single source of truth for each topic (architecture, CLI, development) to prevent drift
- Provide comprehensive documentation accessible to both agents and humans
- Streamline README.md to quick reference (description, install, basic usage)
- Reduce maintenance burden by eliminating content duplication

**Non-Goals:**
- Duplicate content across multiple files
- Replace existing useful documentation (python-style.md, testing.md, project-structure.md stay)
- Document future/planned features (only current implementation)
- Create Architecture Decision Records (using OpenSpec for this already)
- Over-document for humans (solo project needs essentials with optional deep-dives)

## Decisions

### Decision: Navigation guide instead of comprehensive duplication
Create `docs/agent-guide.md` as a navigation/index that routes to topic-specific documentation, rather than duplicating all content in one place.

**Rationale:**
- Prevents documentation drift from duplicate content
- Single source of truth for each topic (easier maintenance)
- agent-guide.md stays small (~100 lines vs 1000+)
- Comprehensive docs serve both humans and agents
- Lower maintenance burden

**Alternatives considered:**
- One large comprehensive agent-only doc: Rejected due to duplication and drift risk
- No agent guide at all: Rejected because agents need navigation context

### Decision: Topic-specific documentation files
Create separate files for major topics: architecture.md, cli-reference.md, development.md.

**Rationale:**
- Each file has single responsibility and clear scope
- Easy to find and update when changes occur
- Natural fit for both human reading and agent retrieval
- Avoids monolithic documentation files

**Structure:**
- `architecture.md` - System design, modules, data flow, rationale
- `cli-reference.md` - Complete command documentation
- `development.md` - Workflows, testing, logging, code quality
- `agent-guide.md` - Navigation layer with context

### Decision: Lightweight task tracking with markdown files
Create `TODO.md` and `IDEAS.md` in repo root for tracking small tasks and improvement ideas outside the OpenSpec workflow.

**Rationale:**
- Small tasks (< 1 hour) don't warrant full OpenSpec proposals
- Code/architecture reviews generate improvement suggestions that need capture
- Simple markdown files are version controlled and highly visible
- Easy to convert ideas to OpenSpec proposals when ready
- No tooling overhead for solo project

**Structure:**
- `TODO.md` - Small bugs, quick fixes, tasks < 1 hour (checkboxes, priority sections)
- `IDEAS.md` - Improvements from reviews, refactors, optimizations (categorized, with effort/benefit notes)

**Workflow:**
1. Small task identified → add to TODO.md
2. Review generates improvement idea → add to IDEAS.md with context
3. Idea matures → create OpenSpec proposal
4. Task completed → check off or remove from TODO.md

### Decision: Maintenance through OpenSpec integration
Require documentation updates as part of OpenSpec change proposals that affect architecture, CLI, or workflows.

**Rationale:**
- Ensures documentation stays synchronized
- Leverages existing change management process
- Makes documentation a first-class concern

**Implementation:**
- Add reminder in AGENTS.md to update agent-guide.md for relevant changes
- Include documentation updates in task lists for proposals affecting documented areas

### Decision: Structured markdown format for all documentation
Use consistent markdown format across all documentation files:
- Consistent heading hierarchy
- Code blocks with language tags
- File paths in `path/to/file.py:line` format
- Command examples with expected output
- Tables for reference data
- Rationale and explanations included where helpful

**Rationale:**
- Both AI and humans benefit from structured, well-formatted documentation
- Code blocks enable accurate command reproduction
- File path format enables direct navigation for both audiences
- Consistency improves readability and retrieval
- "Why" explanations serve as knowledge preservation

### Decision: Streamline README.md, deepen topic docs
README.md provides quick reference only. Topic-specific docs (architecture.md, cli-reference.md, development.md) provide comprehensive information with rationale.

**Rationale:**
- Solo project: maintainer needs quick reference for common tasks
- Comprehensive docs preserve knowledge (prevents "why did I do this?" in 6 months)
- New contributors or agents get full context from topic docs
- README.md serves drive-by users efficiently
- Single source of truth prevents drift

**README.md scope:**
- Project description (2-3 sentences)
- System requirements
- Installation steps
- Quick start examples
- Links to detailed documentation

**Topic docs scope:**
- Complete information on the topic
- Design rationale and trade-offs
- Examples with explanations
- File locations and module structure

## File Structure

```
docs/
├── agent-guide.md           # Navigation/index for AI agents (~100 lines)
│   ├── Quick Reference by Task
│   ├── Documentation Map
│   └── Links to detailed docs with context
│
├── architecture.md          # System design and rationale (NEW)
│   ├── Overview
│   ├── Module Organization
│   ├── Data Flow
│   ├── Key Design Decisions
│   └── Tech Stack Choices
│
├── cli-reference.md         # Complete CLI documentation (NEW)
│   ├── Command Structure
│   ├── analyze Command (all options)
│   ├── evaluate Command (all options)
│   ├── Configuration
│   └── Examples with Output
│
├── development.md           # Development workflows (NEW)
│   ├── Testing
│   ├── Logging Patterns
│   ├── Code Quality Tools
│   ├── Evaluation Framework
│   └── Common Development Tasks
│
├── python-style.md          # Existing - coding standards
├── testing.md               # Existing - testing practices
└── project-structure.md     # Existing - directory layout

TODO.md                      # Small tasks, quick fixes (< 1 hour) (NEW)
IDEAS.md                     # Improvement ideas, potential proposals (NEW)
README.md                    # Streamlined quick reference
CLAUDE.md                    # Agent instructions + links + task management
openspec/project.md          # Future vision, planned architecture
```

## agent-guide.md Structure (Navigation Only)

```markdown
# AI Agent Guide to EDM Project

Navigation guide for AI agents. Links to detailed documentation with context.

## Quick Reference by Task

### Installing and Running
- Installation: See README.md
- Why uv? [brief context]
- CLI: See cli-reference.md

### Understanding Architecture
- System overview: See architecture.md
- Module locations: [list with file paths]
- Design rationale: See architecture.md#decisions

### Running Analysis
- Commands: See cli-reference.md#analyze
- BPM detection flow: [brief explanation + link to architecture.md]
- Implementation: [file paths]
- Configuration: [brief + link]

### Running Evaluations
- Commands: See cli-reference.md#evaluate
- Purpose: [brief explanation]
- Results: [location + link to development.md]

### Development Workflows
- Testing: See testing.md and development.md#testing
- Logging: See development.md#logging
- Code quality: See python-style.md and development.md

### Finding Code
- By feature: [file paths]
- Project structure: See project-structure.md

### Making Changes
- OpenSpec: See openspec/AGENTS.md
- Update docs: [which files to update when]

## Documentation Map
[Table of documents with purpose and when to read]
```

## Risks / Trade-offs

**Risk:** Documentation drift as code evolves
- **Mitigation:** Integrate updates into OpenSpec workflow
- **Mitigation:** Single source of truth per topic reduces drift surface area
- **Future:** Add CI validation for file paths and code examples

**Risk:** Agents may not discover navigation guide
- **Mitigation:** Link prominently from CLAUDE.md
- **Mitigation:** agent-guide.md provides clear value (saves search time)

**Trade-off:** Multiple topic files vs one comprehensive file
- **Pro:** Single source of truth, easier maintenance, serves both audiences
- **Con:** Agents must read multiple files for complete picture
- **Decision:** Navigation guide (agent-guide.md) provides efficient routing

**Trade-off:** README.md streamlining
- **Pro:** Faster for quick reference, less clutter
- **Con:** May lose some helpful context
- **Decision:** Move context to topic docs where it's more discoverable

## Migration Plan

### Phase 1: Create Topic Documentation
1. Create `docs/architecture.md` - consolidate architecture info from openspec/project.md and README.md
2. Create `docs/cli-reference.md` - extract CLI details from README.md, add comprehensive options
3. Create `docs/development.md` - consolidate workflow info from CLAUDE.md and scattered sources

### Phase 2: Create Navigation Layer
4. Create `docs/agent-guide.md` - navigation with links to topic docs
5. Add link in CLAUDE.md to agent-guide.md

### Phase 3: Streamline README
6. Streamline README.md to essentials (description, install, quick start, links)
7. Ensure all removed content exists in topic docs

### Validation
- Verify all file paths in documentation are accurate
- Verify all code examples are valid
- Test agent can complete common tasks using only documentation
- Verify no content duplication across files

## Open Questions
None - approach is clear with single source of truth per topic.
