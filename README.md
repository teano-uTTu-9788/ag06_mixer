# Terminal Automation Framework

A comprehensive, modular shell framework following Google/Meta engineering best practices for macOS development environments.

## 🚀 Quick Start

```bash
# Make CLI executable
chmod +x dev

# Check system health
./dev doctor

# Bootstrap development environment
./dev bootstrap

# Run all quality checks
./dev ci

# Get help
./dev help
```

## 📋 Features

### Core Commands
- **`bootstrap`** - Setup complete development environment with Homebrew
- **`doctor`** - Comprehensive system health checks and diagnostics
- **`build`** - Build projects (Python, Node.js, Docker)
- **`test`** - Run comprehensive test suites
- **`lint`** - Code quality checks with shellcheck, ruff, etc.
- **`fmt`** - Automatic code formatting
- **`ci`** - Run all CI checks locally
- **`deploy`** - Deploy to staging/production environments
- **`clean`** - Clean build artifacts and caches
- **`security`** - Run security scans and vulnerability checks
- **`shell`** - Enter development shell with environment loaded

### Architecture

```
terminal-automation/
├── dev                          # Main CLI entry point
├── scripts/
│   ├── core.sh                  # Core framework functions
│   └── lib/                     # Modular libraries
│       ├── colors.sh           # Terminal colors/formatting
│       ├── logging.sh          # Structured logging
│       ├── validation.sh       # Input/system validation
│       ├── homebrew.sh         # macOS package management
│       ├── git.sh              # Git operations
│       └── testing.sh          # Testing utilities
├── .github/workflows/          # CI/CD automation
│   └── ci.yml                  # GitHub Actions pipeline
└── Brewfile                    # Homebrew dependencies
```

## 🔧 Design Principles

Following industry best practices from leading tech companies:

### Google Shell Style Guide Compliance
- **Single Purpose**: Each script focused on one task
- **Modular Design**: Reusable library components
- **Error Handling**: Comprehensive validation and logging
- **Documentation**: Self-documenting code

### Meta-Inspired Automation
- **Gradual Rollout**: Test → Stage → Production patterns
- **Comprehensive Testing**: Unit, integration, system tests
- **Monitoring Integration**: Health checks and observability

### Netflix Quality Patterns
- **Test-Driven**: Automated testing and quality gates
- **Performance Focus**: Parallel execution, caching
- **Developer Experience**: Fast onboarding, minimal friction

## 📦 Installation

### Prerequisites
- macOS (Intel or Apple Silicon)
- Bash 3.2+ (macOS default) or newer

### Install Framework
```bash
# Clone or download framework
git clone <repository-url>
cd terminal-automation-framework

# Make CLI executable
chmod +x dev

# Bootstrap environment
./dev bootstrap
```

### Dependencies
The framework will automatically install required tools via Homebrew:
- Git, curl, wget, jq
- shellcheck, shfmt (shell development)
- GitHub CLI (gh)
- Security tools (gitleaks, bandit)
- And more (see Brewfile)

## 🏗️ Usage Examples

### Development Workflow
```bash
# Setup new project
./dev bootstrap

# Check everything is working
./dev doctor

# Run tests and linting
./dev ci

# Deploy to staging
./dev deploy staging
```

### Continuous Integration
The framework includes GitHub Actions workflows that:
- Validate framework structure
- Run quality gates (linting, security)
- Execute comprehensive tests
- Build and package artifacts

### Custom Commands
Extend the framework by adding functions to `dev` file:

```bash
cmd::my_custom_command() {
    log::info "Running custom command..."
    # Your implementation here
}
```

## 🧪 Testing

```bash
# Run all tests
./dev test

# Run specific test types
./dev test unit
./dev test integration

# Check code quality
./dev lint
./dev security
```

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
- **Validation**: Framework structure and syntax
- **Quality Gates**: Linting, formatting, security
- **Testing**: Multi-platform test execution
- **Build**: Package creation and artifact upload
- **Deploy**: Automated deployment to environments

## 🛠️ Customization

### Adding Libraries
Create new library in `scripts/lib/`:

```bash
# scripts/lib/my_lib.sh
my_lib::function() {
    log::info "Custom library function"
}
```

Load in main script:
```bash
deps::require "my_lib"
```

### Environment Variables
- `FRAMEWORK_DEBUG=true` - Enable debug logging
- `LOG_LEVEL` - Set logging level
- `AUTOMATION_LIB_DIR` - Override library directory

## 📊 Monitoring & Observability

- Structured logging with timestamps
- Health checks and system diagnostics
- Performance monitoring
- Error tracking and reporting

## 🔒 Security

- Secret scanning with gitleaks
- Shell script security validation
- Input validation and sanitization
- Secure defaults throughout

## 🤝 Contributing

1. Follow Google Shell Style Guide
2. Add tests for new functionality
3. Update documentation
4. Run `./dev ci` before submitting

## 📄 License

[Add your license here]

## 🙋 Support

For issues and questions:
1. Check `./dev doctor` output
2. Review logs in debug mode: `FRAMEWORK_DEBUG=true ./dev <command>`
3. Submit issues with full error output

---

Built with ❤️ following industry best practices from Google, Meta, and Netflix.