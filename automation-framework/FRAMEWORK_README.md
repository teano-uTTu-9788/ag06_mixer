# Terminal Automation Framework

A modular, CI/CD-compatible terminal automation framework for macOS, following best practices from Google and Meta. This framework provides a unified developer CLI for managing Homebrew packages, Git operations, code quality tools, and automated workflows.

## 🎯 Key Features

- **Google Shell Style Guide Compliance**: All shell scripts follow Google's recommended practices
- **Meta-Style CI/CD Integration**: GitHub Actions workflows with matrix testing and fail-fast behavior
- **Modular Architecture**: Reusable library functions for common operations
- **Developer Experience Focus**: Unified CLI interface with comprehensive help and error handling
- **Homebrew Integration**: Automated package management with dependency tracking
- **Code Quality Tools**: Integrated ShellCheck linting and shfmt formatting
- **Comprehensive Testing**: BATS test framework with extensive validation

## 🚀 Quick Start

```bash
# Make the dev script executable
chmod +x ./dev

# Bootstrap your development environment
./dev bootstrap

# Check system health
./dev doctor

# View all available commands
./dev help
```

## 📋 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `doctor` | Check system health and dependencies | `./dev doctor` |
| `bootstrap` | Bootstrap development environment | `./dev bootstrap` |
| `install <pkg>` | Install package via Homebrew | `./dev install shellcheck` |
| `update` | Update Homebrew and all packages | `./dev update` |
| `cleanup` | Clean up Homebrew cache | `./dev cleanup` |
| `git:setup <name> <email>` | Configure Git with recommended settings | `./dev git:setup "Your Name" "email@example.com"` |
| `git:branch <name>` | Create and switch to new branch | `./dev git:branch feature/new-feature` |
| `format` | Format shell scripts with shfmt | `./dev format` |
| `lint` | Run ShellCheck on all scripts | `./dev lint` |
| `test` | Run BATS test suite | `./dev test` |
| `ci` | Run full CI pipeline locally | `./dev ci` |

## 🏗️ Architecture

### Directory Structure

```
automation-framework/
├── dev                     # Main CLI entry point
├── scripts/
│   └── lib/
│       ├── core.sh         # Core utilities and logging
│       ├── homebrew.sh     # Homebrew automation
│       └── git.sh          # Git operations
├── tests/
│   └── framework.bats      # BATS test suite
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI/CD
├── Brewfile                # Homebrew dependencies
└── FRAMEWORK_README.md
```

### Core Principles

1. **Modularity**: Each library handles a specific domain (Git, Homebrew, etc.)
2. **Error Handling**: Comprehensive error checking with proper cleanup
3. **Logging**: Structured logging with configurable levels
4. **Validation**: Input validation and command verification
5. **Performance**: Timing and retry logic for reliability

## 🧪 Testing

### Running Tests

```bash
# Install BATS testing framework
brew install bats-core

# Run the complete test suite
./dev test

# Run tests with comprehensive checks (slower)
BATS_COMPREHENSIVE_TESTS=1 ./dev test

# Run specific test file
bats tests/framework.bats
```

### Test Categories

- **Basic Functionality**: CLI interface and command parsing
- **System Health**: Environment validation and dependency checking
- **Integration**: Homebrew, Git, and package management operations
- **Code Quality**: Linting and formatting tool integration
- **Error Handling**: Graceful degradation and recovery
- **Performance**: Command execution timing and reliability

## 🔄 CI/CD Integration

### GitHub Actions

The framework includes a comprehensive GitHub Actions workflow with:

- **Matrix Testing**: Multiple macOS versions (13, 14, latest)
- **Dependency Caching**: Homebrew package caching for faster builds
- **Quality Checks**: ShellCheck linting and formatting validation
- **Integration Tests**: End-to-end testing of all framework components
- **Fail-Fast**: Quick feedback with parallel job execution

### Local CI Pipeline

Run the complete CI pipeline locally:

```bash
./dev ci
```

This runs:
1. System health check
2. Code formatting validation
3. ShellCheck linting
4. Test suite execution
5. Git status validation

## 🛠️ Development

### Adding New Commands

1. Add the command function to the `dev` script:
```bash
cmd_new_command() {
  local param="$1"
  log_info "Executing new command with: ${param}"
  # Implementation here
}
```

2. Add the command to the `main()` function's case statement
3. Update the `usage()` function with command documentation
4. Add tests to `tests/framework.bats`

### Adding New Libraries

1. Create new library file in `scripts/lib/`
2. Follow the existing pattern with function exports
3. Source the library in the main `dev` script
4. Add validation tests

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARN, ERROR) | `INFO` |
| `PROJECT_ROOT` | Override project root directory | Auto-detected |
| `CI` | CI environment detection | `false` |

## 📚 Best Practices Applied

### Google Practices
- **Shell Style Guide**: Strict adherence to Google Shell Style Guide
- **Error Handling**: Comprehensive error checking with `set -euo pipefail`
- **Logging**: Structured logging with timestamps and levels
- **Function Organization**: Single-purpose functions with clear responsibilities

### Meta Practices  
- **Developer Experience**: Unified CLI with consistent interface
- **CI/CD Pipeline**: Matrix testing with fail-fast behavior
- **Performance**: Timing and caching optimizations
- **Reliability**: Retry logic and graceful degradation

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`./dev git:branch feature/amazing-feature`)
3. **Commit** your changes following conventional commits
4. **Test** your changes (`./dev ci`)
5. **Push** to the branch
6. **Open** a Pull Request

## 📄 License

This project follows open-source best practices. Please see LICENSE file for details.

## 🔗 Related Resources

- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- [Meta Engineering Blog](https://engineering.fb.com/)
- [Homebrew Documentation](https://docs.brew.sh/)
- [BATS Testing Framework](https://github.com/bats-core/bats-core)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

Built with ❤️ following industry best practices from Google, Meta, and the broader developer community.