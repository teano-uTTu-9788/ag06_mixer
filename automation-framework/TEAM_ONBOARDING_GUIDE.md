# Terminal Automation Framework - Team Onboarding Guide

Welcome to the Terminal Automation Framework! This guide will help you get started quickly and effectively.

---

## 🎯 What is the Terminal Automation Framework?

A unified developer CLI tool that standardizes common development tasks across your team, following best practices from Google and Meta. It provides:

- **Single entry point** (`./dev`) for all developer operations
- **Consistent commands** across different projects and environments
- **Automated workflows** for CI/CD, testing, and deployment
- **Best practices** enforcement through linting and formatting

---

## 🚀 Quick Start (5 minutes)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd automation-framework
```

### Step 2: Run Initial Setup
```bash
# Make the dev command executable
chmod +x dev

# Check your system health
./dev doctor

# Bootstrap your environment (installs tools)
./dev bootstrap
```

### Step 3: Try Basic Commands
```bash
# Show available commands
./dev help

# Check framework version
./dev version

# Run the test suite
./dev test

# Run local CI pipeline
./dev ci
```

---

## 📋 Essential Commands

### Daily Development
| Command | Purpose | Example |
|---------|---------|---------|
| `dev doctor` | Check system health | `./dev doctor` |
| `dev lint` | Check code quality | `./dev lint` |
| `dev format` | Format shell scripts | `./dev format` |
| `dev test` | Run test suite | `./dev test` |
| `dev ci` | Run full CI locally | `./dev ci` |

### Environment Management
| Command | Purpose | Example |
|---------|---------|---------|
| `dev bootstrap` | Setup environment | `./dev bootstrap` |
| `dev install <pkg>` | Install package | `./dev install jq` |
| `dev update` | Update all packages | `./dev update` |
| `dev cleanup` | Clean cache | `./dev cleanup` |

### Git Operations
| Command | Purpose | Example |
|---------|---------|---------|
| `dev git:setup` | Configure Git | `./dev git:setup "John Doe" "john@example.com"` |
| `dev git:branch` | Create branch | `./dev git:branch feature/new-feature` |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Homebrew Permission Error
**Symptom**: `Permission denied @ dir_s_mkdir - /private/tmp`

**Solution**:
```bash
# Run the fix script (requires sudo)
./fix_homebrew_permissions.sh
```

#### 2. Missing Development Tools
**Symptom**: `shellcheck`, `shfmt`, or `bats` not found

**Solution**:
```bash
# Run the tool installation script
./install_dev_tools.sh

# Add ~/bin to PATH if needed
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 3. Command Not Found
**Symptom**: `./dev: command not found`

**Solution**:
```bash
# Make sure you're in the right directory
pwd  # Should show .../automation-framework

# Make dev executable
chmod +x dev
```

---

## 🏗️ Framework Architecture

### Directory Structure
```
automation-framework/
├── dev                    # Main CLI entry point
├── scripts/lib/           # Modular libraries
│   ├── core.sh           # Logging, validation
│   ├── homebrew.sh       # Package management
│   └── git.sh            # Git operations
├── tests/                 # Test suites
├── .github/workflows/     # CI/CD pipelines
└── docs/                  # Documentation
```

### How It Works
1. **`dev` command** receives user input
2. **Libraries** provide reusable functions
3. **Commands** execute specific operations
4. **Logging** tracks all activities
5. **Error handling** ensures graceful failures

---

## 👥 Team Workflows

### Feature Development
```bash
# 1. Create feature branch
./dev git:branch feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Format and lint code
./dev format
./dev lint

# 4. Run tests
./dev test

# 5. Run CI locally before pushing
./dev ci

# 6. Commit and push
git add .
git commit -m "feat: Add new feature"
git push origin feature/my-feature
```

### Code Review Preparation
```bash
# Ensure code quality before PR
./dev format    # Format code
./dev lint      # Check for issues
./dev test      # Run tests
./dev ci        # Full CI check
```

### Debugging Issues
```bash
# Check system status
./dev doctor

# Run with debug logging
LOG_LEVEL=DEBUG ./dev <command>

# Check logs
tail -f automation.log
```

---

## 📚 Best Practices

### 1. Always Run CI Locally
Before pushing code, run `./dev ci` to catch issues early.

### 2. Use Consistent Branch Names
Follow the pattern: `type/description`
- `feature/add-login`
- `bugfix/fix-memory-leak`
- `chore/update-deps`

### 3. Keep Tools Updated
Run `./dev update` weekly to keep dependencies current.

### 4. Document Custom Commands
If you add new commands, update the help text and documentation.

---

## 🔐 Security Guidelines

### Do's
- ✅ Use `./dev git:setup` for consistent Git configuration
- ✅ Run `./dev lint` to catch security issues
- ✅ Keep dependencies updated with `./dev update`

### Don'ts
- ❌ Don't commit sensitive data (keys, passwords)
- ❌ Don't skip CI checks before merging
- ❌ Don't modify core libraries without team review

---

## 📈 Advanced Usage

### Custom Commands
Add new commands by editing the `dev` script:

```bash
# In dev script
cmd_mycommand() {
    log_info "Running my custom command..."
    # Your logic here
}

# In main() function
mycommand)
    cmd_mycommand "$@"
    ;;
```

### Environment Variables
```bash
# Enable debug logging
LOG_LEVEL=DEBUG ./dev doctor

# Run in CI mode
CI=true ./dev test

# Custom project root
PROJECT_ROOT=/custom/path ./dev lint
```

### Extending Libraries
Create new libraries in `scripts/lib/`:
```bash
# scripts/lib/custom.sh
custom_function() {
    log_info "Custom function called"
}

# Source in dev script
source "$LIB_DIR/custom.sh"
```

---

## 🆘 Getting Help

### Resources
1. **Framework README**: `FRAMEWORK_README.md`
2. **Issues & Improvements**: `ISSUES_AND_IMPROVEMENTS.md`
3. **Deployment Guide**: `DEPLOYMENT_SUMMARY.md`

### Support Channels
- **GitHub Issues**: Report bugs and request features
- **Team Chat**: #dev-automation channel
- **Documentation**: Check the `docs/` directory

### Quick Diagnostics
```bash
# Full system check
./dev doctor

# Version information
./dev version

# Help for any command
./dev help
```

---

## 🎓 Learning Path

### Week 1: Basics
- [ ] Run `./dev doctor` and fix any issues
- [ ] Complete bootstrap with `./dev bootstrap`
- [ ] Try all basic commands from Quick Start
- [ ] Read through available documentation

### Week 2: Integration
- [ ] Create a feature branch with `./dev git:branch`
- [ ] Run full CI pipeline with `./dev ci`
- [ ] Fix any linting issues found
- [ ] Submit first PR using the framework

### Week 3: Advanced
- [ ] Add a custom command
- [ ] Contribute to documentation
- [ ] Help onboard another team member
- [ ] Suggest improvements

---

## ✅ Onboarding Checklist

Complete these steps to be fully onboarded:

- [ ] Repository cloned
- [ ] `./dev doctor` shows all green
- [ ] Bootstrap completed successfully
- [ ] All essential commands tested
- [ ] First CI pipeline run locally
- [ ] Documentation reviewed
- [ ] First PR submitted using framework
- [ ] Added to team communication channels

---

## 🎉 Welcome to the Team!

You're now ready to use the Terminal Automation Framework. Remember:
- **Ask questions** - We're here to help
- **Suggest improvements** - Your feedback matters
- **Share knowledge** - Help others learn

Happy coding! 🚀

---

*Terminal Automation Framework v2.0.0 - Built with ❤️ by the team*