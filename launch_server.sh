#!/bin/bash
# aioke_server_launcher.sh
# One-click launcher for the AiOke Automation Framework Backend

# Text Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   AiOke Studio: Backend Launcher      ${NC}"
echo -e "${BLUE}=======================================${NC}"

# 1. Check for Python/Pip
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

# 2. Check for dependencies (simple check)
if python3 -c "import flask, flask_cors, sounddevice" &> /dev/null; then
    echo -e "${GREEN}[OK] Dependencies found (Flask, SoundDevice).${NC}"
else
    echo -e "${RED}[WARN] Missing dependencies. Installing...${NC}"
    pip3 install flask flask-cors sounddevice numpy
fi

# 3. Get Local IP Address (Mac/Linux)
IP_ADDRESS=$(ipconfig getifaddr en0 || echo "127.0.0.1")

echo -e "\n${GREEN}Server launching...${NC}"
echo -e "------------------------------------------------"
echo -e "📱  ${BLUE}INSTRUCTIONS FOR MOBILE APP:${NC}"
echo -e "1. Open AiOke on your iPhone."
echo -e "2. In the 'Server URL' box, enter:"
echo -e "   ${GREEN}http://$IP_ADDRESS:8899${NC}"
echo -e "------------------------------------------------"
echo -e "Press Ctrl+C to stop the server.\n"

# 4. Run the Server
# Assuming this script is run from project root, look for app.py
if [ -f "automation-framework/app.py" ]; then
    python3 automation-framework/app.py
elif [ -f "app.py" ]; then
    python3 app.py
else
    echo -e "${RED}[ERROR] Cannot find app.py! Run this from the project root.${NC}"
    exit 1
fi
