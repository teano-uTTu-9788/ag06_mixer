# macOS Terminal Automation Framework

A CI/CD-compatible, modular terminal automation framework tailored for macOS using Homebrew tools, following best practices from Google, Meta, and leading tech companies.

## 🏗️ Architecture

```
automation-framework/
├── bin/                    # CLI entry points
│   ├── dev                # Main developer CLI
│   └── dev-completion     # Shell completions
├── lib/                   # Reusable libraries
│   ├── core/             # Core utilities
│   │   ├── logger.sh     # Logging functions
│   │   ├── colors.sh     # Color output
│   │   ├── utils.sh      # Common utilities
│   │   └── validation.sh # Input validation
│   ├── homebrew/         # Homebrew management
│   │   ├── install.sh    # Package installation
│   │   └── bundle.sh     # Brewfile management
│   ├── git/              # Git automation
│   │   ├── hooks.sh      # Git hooks
│   │   └── workflow.sh   # Git workflows
│   └── ci/               # CI/CD utilities
│       ├── github.sh     # GitHub Actions helpers
│       └── docker.sh     # Docker utilities
├── scripts/              # Standalone scripts
│   ├── setup/           # Setup scripts
│   │   ├── macos.sh    # macOS configuration
│   │   └── dev-env.sh  # Development environment
│   ├── maintenance/     # Maintenance scripts
│   │   ├── cleanup.sh  # System cleanup
│   │   └── update.sh   # Update all tools
│   └── deploy/         # Deployment scripts
│       └── release.sh  # Release automation
├── config/             # Configuration files
│   ├── defaults.conf   # Default settings
│   └── Brewfile       # Homebrew dependencies
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── .github/           # GitHub configuration
│   └── workflows/     # GitHub Actions
│       ├── ci.yml     # CI pipeline
│       └── release.yml # Release pipeline
└── docs/              # Documentation
    ├── ARCHITECTURE.md
    ├── CONTRIBUTING.md
    └── API.md
```

## 🚀 Features

- **Modular Design**: Following Google's principle of small, focused utilities
- **Homebrew Integration**: Centralized package management
- **CI/CD Ready**: GitHub Actions workflows included
- **Developer CLI**: Unified entry point for all automation
- **Shell Best Practices**: Following Google Shell Style Guide
- **Testing Framework**: Unit and integration tests
- **Documentation**: Comprehensive docs and examples

## 📋 Prerequisites

- macOS 11.0 or later
- Homebrew installed
- Bash 4.0+ (installed via Homebrew)
- Git 2.0+

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automation-framework.git
cd automation-framework

# Run setup
./bin/dev setup

# Add to PATH
echo 'export PATH="$HOME/automation-framework/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Enable completions
dev completions install
```

## 💻 Usage

```bash
# Main CLI
dev <command> [options]

# Available commands:
dev setup              # Initial setup
dev update             # Update all tools
dev doctor             # Check system health
dev clean              # Clean temporary files
dev test               # Run test suite
dev deploy             # Deploy application
dev ci                 # Run CI checks locally
```

## 🔧 Configuration

Create a local configuration file:

```bash
cp config/defaults.conf config/local.conf
```

## 🧪 Testing

```bash
# Run all tests
dev test

# Run specific test suite
dev test unit
dev test integration

# Run with coverage
dev test --coverage
```

## 📚 Best Practices Applied

### From Google:
- Bash-only for consistency
- Scripts under 100 lines
- Extensive error handling
- Security-first approach
- Comprehensive documentation

### From Meta:
- "Move Fast" with automation
- Abstract complexity
- Automate repetitive tasks
- Scale through tooling
- Continuous deployment

### From Industry Leaders:
- Infrastructure as Code
- GitOps workflows
- Observable systems
- Incremental rollouts
- Self-healing systems

## 🤝 Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.