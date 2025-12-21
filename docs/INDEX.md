# EDM Documentation

**Welcome to the EDM documentation.** This page helps you find the right documentation quickly.

## Quick Navigation

### New to EDM?
- **[Getting Started Guide](getting-started.md)** - Installation, setup, and first commands
- **[Architecture Overview](reference/architecture.md)** - Understand the system design
- **[Terminology](reference/terminology.md)** - EDM-specific terms and concepts

### I want to...

#### Train a Model
- **[Training Guide](guides/training.md)** - Complete training reference
- **[Training Cheatsheet](cheatsheets/training.md)** - Quick command reference
- **[Model Management](guides/model-management.md)** - Experiment tracking and versioning

#### Annotate Tracks
- **[Annotator Guide](guides/annotator.md)** - Web-based annotation tool
- Setup, development, deployment, and troubleshooting

#### Analyze Audio
- **[CLI Reference](reference/cli.md)** - All command-line commands
- **[Common Commands](cheatsheets/common-commands.md)** - Quick reference cheatsheet

#### Deploy to Production
- **[Deployment Guide](deployment.md)** - Production deployment strategies
- **[Troubleshooting](reference/troubleshooting.md)** - Common issues and solutions

#### Work with Data
- **[Data Management](guides/data-management.md)** - Annotations, import/export, validation

#### Debug Issues
- **[Troubleshooting Guide](reference/troubleshooting.md)** - Organized by component
- Training → [Training Guide - Troubleshooting](guides/training.md#troubleshooting)
- Annotator → [Annotator Guide - Troubleshooting](guides/annotator.md#troubleshooting)

## Documentation by Role

### Users
- [Getting Started](getting-started.md)
- [CLI Reference](reference/cli.md)
- [Common Commands Cheatsheet](cheatsheets/common-commands.md)
- [Troubleshooting](reference/troubleshooting.md)

### Researchers / ML Engineers
- [Training Guide](guides/training.md)
- [Model Management](guides/model-management.md)
- [Data Management](guides/data-management.md)
- [Architecture](reference/architecture.md)

### Developers / Contributors
- [Development Setup](development/setup.md)
- [Testing Guide](development/testing.md)
- [Python Style Guide](development/code-style-python.md)
- [JavaScript Style Guide](development/code-style-javascript.md)
- [Agent Guide](agent-guide.md) - For AI assistants

### DevOps / Operators
- [Deployment Guide](deployment.md)
- [Troubleshooting](reference/troubleshooting.md)
- [Architecture](reference/architecture.md)

## Documentation Structure

```
docs/
├── INDEX.md (this file)          # Documentation hub
├── getting-started.md            # Setup and first steps
│
├── guides/                       # Topic-based guides
│   ├── training.md               # Complete training reference
│   ├── annotator.md              # Annotation tool guide
│   ├── model-management.md       # Experiment tracking & MLflow
│   └── data-management.md        # Data formats and workflows
│
├── reference/                    # Look-up documentation
│   ├── cli.md                    # CLI command reference
│   ├── architecture.md           # System design
│   ├── project-structure.md      # Directory layout
│   ├── terminology.md            # Glossary
│   └── troubleshooting.md        # Problem solutions
│
├── development/                  # Contributor documentation
│   ├── setup.md                  # Development environment
│   ├── testing.md                # Test framework
│   ├── code-style-python.md      # Python conventions
│   ├── code-style-javascript.md  # JavaScript/TypeScript conventions
│   └── claude-integration.md     # Claude Code plugins
│
├── cheatsheets/                  # Quick references
│   ├── training.md               # Training commands
│   └── common-commands.md        # Frequently used commands
│
├── deployment.md                 # Production deployment
└── agent-guide.md                # AI assistant navigation
```

## Finding Specific Information

### Command Reference
**"What's the syntax for...?"** → [CLI Reference](reference/cli.md) or [Common Commands](cheatsheets/common-commands.md)

### Troubleshooting
**"Why isn't X working?"** → [Troubleshooting Guide](reference/troubleshooting.md)

Quick links:
- Installation issues → [Troubleshooting - Installation](reference/troubleshooting.md#installation-and-setup)
- Training issues → [Troubleshooting - Training](reference/troubleshooting.md#training)
- Annotator issues → [Troubleshooting - Annotator](reference/troubleshooting.md#annotator-application)

### How-To Guides
**"How do I...?"**
- Train a model → [Training Guide](guides/training.md)
- Set up dev environment → [Development Setup](development/setup.md)
- Deploy to production → [Deployment Guide](deployment.md)
- Manage experiments → [Model Management](guides/model-management.md)

### Concepts & Design
**"How does X work?"** → [Architecture](reference/architecture.md)

### Code Style
**"What's the convention for...?"**
- Python → [Python Style Guide](development/code-style-python.md)
- JavaScript/TypeScript → [JavaScript Style Guide](development/code-style-javascript.md)

## External Resources

- **Repository**: [github.com/crsmithdev/edm](https://github.com/crsmithdev/edm)
- **Issues**: [Report bugs or request features](https://github.com/crsmithdev/edm/issues)
- **Packages**:
  - [edm-lib](../packages/edm-lib/README.md) - Core library
  - [edm-cli](../packages/edm-cli/README.md) - Command-line interface
  - [edm-annotator](../packages/edm-annotator/README.md) - Web annotation tool

## Contributing to Documentation

Found an issue or want to improve the docs?

1. Documentation source is in `docs/` directory
2. Follow the structure outlined above
3. Use Markdown format
4. Cross-reference related docs with relative links
5. Submit a pull request

See [Development Setup](development/setup.md) for contribution guidelines.

---

**Can't find what you're looking for?** Check the [Troubleshooting Guide](reference/troubleshooting.md#getting-help) or open an issue.
