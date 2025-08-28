#!/usr/bin/env bash
# Universal Parallel Workflow System Deployment Script
# Deploys the 88/88 compliant universal workflow orchestration system

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 UNIVERSAL PARALLEL WORKFLOW SYSTEM DEPLOYMENT${NC}"
echo "=================================================="
echo

# Check if we're in the right directory
if [[ ! -f "universal_parallel_orchestrator.sh" ]]; then
    echo -e "${RED}❌ Error: Must run from automation-framework directory${NC}"
    exit 1
fi

# Step 1: Verify all required files exist
echo -e "${YELLOW}📋 Step 1: Verifying required files...${NC}"
required_files=(
    "universal_parallel_orchestrator.sh"
    "parallel_workflow_orchestrator.sh"
    "dev"
    "critical_assessment_universal_workflow_88.py"
    "final_88_test_validation.py"
    "test_universal_workflow_validation.py"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "  ✅ $file"
    else
        echo -e "  ${RED}❌ Missing: $file${NC}"
        exit 1
    fi
done

# Step 2: Make scripts executable
echo -e "\n${YELLOW}📋 Step 2: Setting executable permissions...${NC}"
chmod +x universal_parallel_orchestrator.sh
chmod +x parallel_workflow_orchestrator.sh
chmod +x dev
echo -e "  ✅ Scripts are executable"

# Step 3: Initialize universal workflow system
echo -e "\n${YELLOW}📋 Step 3: Initializing universal workflow environment...${NC}"
./dev universal:init
echo -e "  ✅ Universal environment initialized"

# Step 4: Analyze current project
echo -e "\n${YELLOW}📋 Step 4: Analyzing project and creating tasks...${NC}"
./dev universal:analyze
echo -e "  ✅ Project analyzed and tasks created"

# Step 5: Run validation tests
echo -e "\n${YELLOW}📋 Step 5: Running 88-test validation suite...${NC}"
if python3 final_88_test_validation.py; then
    echo -e "  ${GREEN}✅ 88/88 tests passed - System fully compliant${NC}"
else
    echo -e "  ${RED}⚠️ Some tests failed - Review output above${NC}"
fi

# Step 6: Create symbolic links for global access (optional)
echo -e "\n${YELLOW}📋 Step 6: Creating global access (optional)...${NC}"
read -p "Install dev CLI globally in ~/bin? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p ~/bin
    ln -sf "$(pwd)/dev" ~/bin/dev-universal
    echo -e "  ✅ Created ~/bin/dev-universal symlink"
    echo -e "  ${BLUE}ℹ️  Add ~/bin to your PATH to use 'dev-universal' globally${NC}"
else
    echo -e "  ⏭️  Skipped global installation"
fi

# Step 7: Display usage information
echo -e "\n${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo
echo -e "${BLUE}📚 USAGE GUIDE:${NC}"
echo "================================"
echo
echo "Universal Commands (work with any repository):"
echo "  ./dev universal:init       - Initialize workflow environment"
echo "  ./dev universal:register   - Register a Claude instance"
echo "  ./dev universal:analyze    - Analyze project and create tasks"
echo "  ./dev universal:status     - View current workflow status"
echo "  ./dev universal:distribute - Distribute tasks to instances"
echo "  ./dev universal:monitor    - Monitor workflow progress"
echo
echo "Parallel Commands (AiOke-specific legacy):"
echo "  ./dev parallel:init        - Initialize AiOke workflow"
echo "  ./dev parallel:status      - View AiOke workflow status"
echo "  ./dev parallel:distribute  - Distribute AiOke tasks"
echo
echo -e "${BLUE}🎯 QUICK START:${NC}"
echo "1. Register instances:"
echo "   ./dev universal:register instance1 backend_development \"Working on backend\""
echo "   ./dev universal:register instance2 frontend_development \"Working on UI\""
echo
echo "2. Distribute tasks:"
echo "   ./dev universal:distribute"
echo
echo "3. Monitor progress:"
echo "   ./dev universal:monitor"
echo
echo -e "${GREEN}🎉 Universal Parallel Workflow System is ready!${NC}"
echo "Multiple Claude instances can now collaborate on ANY repository!"