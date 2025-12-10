# Claude Code Plugins and Skills

This document lists recommended Claude Code plugins and skills for working with the EDM project, particularly for JavaScript/TypeScript development.

## Installation

Claude Code plugins can be installed through the CLI or by adding them to your `~/.claude/settings.json` file.

## Recommended JavaScript/TypeScript Plugins

### 1. Claude Code Plugins Plus

A comprehensive plugin hub with 243+ plugins, including 175 with Agent Skills v1.2.0.

**Repository**: [jeremylongshore/claude-code-plugins-plus](https://github.com/jeremylongshore/claude-code-plugins-plus)

**Features**:
- First 100% compliant with Anthropic 2025 Skills schema
- Browse and install plugins from a centralized hub
- Includes skills for JavaScript, TypeScript, React, Node.js, and more

**Installation**:
```bash
# Follow instructions in the repository
# Plugins are typically installed via settings.json
```

### 2. Claud Skills (Production-Ready Framework)

Production-ready framework with 13 agents, 9 skills, and auto-generated documentation for multiple languages.

**Repository**: [Interstellar-code/claud-skills](https://github.com/Interstellar-code/claud-skills)

**Features**:
- JavaScript/TypeScript support
- PHP and Laravel support
- React components and patterns
- Python integration
- Auto-generated documentation
- Production-ready patterns

**Languages Supported**:
- JavaScript
- TypeScript
- PHP
- Laravel
- React
- Python

### 3. Minimal Claude (Auto-Configuration)

Intelligent plugin that automatically configures linting, type checking, and parallel agent-based fixing.

**Repository**: [KenKaiii/minimal-claude](https://github.com/KenKaiii/minimal-claude)

**Features**:
- Auto-configures ESLint
- Auto-configures TypeScript
- Parallel agent-based fixing
- Minimal setup required

### 4. Agents Framework

Intelligent automation and multi-agent orchestration specifically for Claude Code.

**Repository**: [wshobson/agents](https://github.com/wshobson/agents)

**Features**:
- Multi-agent orchestration
- Workflow automation
- Task delegation
- Agent communication patterns

## Official Resources

### Anthropic Claude Agent SDK

Official SDK for building Claude agents with TypeScript support.

**Documentation**: [Claude Docs - Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
**GitHub**: [anthropics/claude-agent-sdk-typescript](https://github.com/anthropics/claude-agent-sdk-typescript)

**Features**:
- Official TypeScript SDK
- Agent Skills support
- Full Claude API integration
- Production-ready patterns

### Claude Code Plugins Marketplace

Official marketplace and CLI plugin manager.

**Website**: [claude-plugins.dev](https://claude-plugins.dev/)

**Features**:
- Browse available plugins
- CLI-based installation
- Plugin ratings and reviews
- Official Anthropic support

## Installing Plugins

### Method 1: Via Settings File

Add plugins to `~/.claude/settings.json`:

```json
{
  "plugins": {
    "enabledPlugins": {
      "plugin-name@source": true
    }
  }
}
```

### Method 2: Via Claude Code CLI

Some plugins provide CLI-based installation. Check the plugin's repository for specific instructions.

### Method 3: Manual Installation

1. Clone the plugin repository to `~/.claude/plugins/`
2. Follow the plugin's installation instructions
3. Enable the plugin in `~/.claude/settings.json`

## Recommended Skills for JavaScript/TypeScript

Based on the plugin ecosystem, here are recommended skills to enable:

### Code Quality
- **ESLint Integration**: Automatic linting and fixing
- **Prettier Integration**: Automatic code formatting
- **TypeScript Checking**: Real-time type checking

### Development Workflow
- **Node.js Helper**: Node.js-specific operations
- **Package Manager**: pnpm/npm/yarn integration
- **Build Tools**: Webpack, Vite, or other bundler support

### Testing
- **Vitest Integration**: Test running and generation
- **Test Coverage**: Coverage analysis
- **E2E Testing**: Playwright or Cypress integration (if applicable)

### Framework-Specific (if using React)
- **React Components**: Component generation
- **React Hooks**: Hook patterns and best practices
- **State Management**: Redux, Zustand, or other state libraries

## Configuration Example

Here's an example `~/.claude/settings.json` configuration with JavaScript/TypeScript plugins:

```json
{
  "model": "claude-sonnet-4-5-20250929",
  "plugins": {
    "enabledPlugins": {
      "javascript-assistant@claude-code-plugins-plus": true,
      "typescript-helper@claude-code-plugins-plus": true,
      "eslint-formatter@minimal-claude": true,
      "prettier-formatter@minimal-claude": true,
      "vitest-runner@claude-code-plugins-plus": true,
      "node-helper@claud-skills": true
    }
  },
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "javascript|typescript|js|ts|jsx|tsx",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/inject-javascript-context.sh"
          }
        ]
      }
    ]
  }
}
```

## Creating a Hook for JavaScript Context

Create `.claude/hooks/inject-javascript-context.sh`:

```bash
#!/bin/bash
# Inject JavaScript context when JS/TS keywords detected

prompt="$1"

if echo "$prompt" | grep -qiE "javascript|typescript|js|ts|jsx|tsx|eslint|prettier|vitest"; then
    cat .claude/contexts/javascript.xml
fi
```

Make it executable:
```bash
chmod +x .claude/hooks/inject-javascript-context.sh
```

## Best Practices

### 1. Start Minimal
Begin with essential plugins (linting, formatting, type checking) and add more as needed.

### 2. Keep Plugins Updated
Regularly check for plugin updates to ensure compatibility with the latest Claude Code version.

### 3. Test Plugin Combinations
Some plugins may conflict. Test combinations in a development environment first.

### 4. Use Plugin-Specific Settings
Many plugins support configuration. Check plugin documentation for available options.

### 5. Monitor Performance
Too many plugins can slow down Claude Code. Disable unused plugins to maintain performance.

## Troubleshooting

### Plugin Not Loading
1. Check `~/.claude/settings.json` syntax
2. Verify plugin is correctly enabled
3. Restart Claude Code session
4. Check plugin logs (if available)

### Plugin Conflicts
1. Disable all plugins
2. Enable one plugin at a time
3. Identify conflicting plugins
4. Check plugin documentation for known conflicts

### Performance Issues
1. Run `/config` to audit configuration
2. Disable unused plugins
3. Check for resource-intensive plugins
4. Consider using lighter alternatives

## Additional Resources

- [Claude Code Documentation](https://www.anthropic.com/news/claude-code-plugins)
- [Agent SDK Overview](https://docs.claude.com/en/api/agent-sdk/overview)
- [Agent Skills Documentation](https://platform.claude.com/docs/en/agent-sdk/skills)
- [Claude Code Plugin Customization](https://www.anthropic.com/news/claude-code-plugins)

## Contributing

To contribute to the plugin ecosystem:

1. Follow the [Anthropic 2025 Skills schema](https://platform.claude.com/docs/en/agent-sdk/skills)
2. Create well-documented plugins
3. Submit to plugin registries
4. Share with the community

## Summary

For the EDM project, we recommend:

1. **Essential**: ESLint, Prettier, TypeScript checking
2. **Testing**: Vitest integration
3. **Development**: Node.js helpers, package manager integration
4. **Optional**: Framework-specific skills (React, if applicable)

Start with the essential plugins and expand based on your workflow needs. The JavaScript context file (`.claude/contexts/javascript.xml`) will automatically inject best practices when working with JS/TS code.
