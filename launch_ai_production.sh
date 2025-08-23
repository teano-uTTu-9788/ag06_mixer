#!/bin/bash
# Production launch script for AG06 AI Mixing Studio
# Uses Gunicorn with gevent for production-grade serving

set -e  # Exit on error

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}🎚️ AG06 AI Mixing Studio - Production Launch${NC}"
echo "=============================================="

# 1. Run port guard
echo -e "${YELLOW}Step 1: Port guard...${NC}"
./port_guard.sh

# 2. Health check dependencies
echo -e "${YELLOW}Step 2: Checking dependencies...${NC}"
python3 -c "import flask, pyaudio, numpy, scipy, gunicorn, gevent" || {
    echo -e "${RED}❌ Missing dependencies. Run: pip install -r requirements.txt${NC}"
    exit 1
}
echo -e "${GREEN}✅ Dependencies OK${NC}"

# 3. Check AG06 is connected
echo -e "${YELLOW}Step 3: Checking AG06...${NC}"
if python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]" | grep -q "AG06"; then
    echo -e "${GREEN}✅ AG06 detected${NC}"
else
    echo -e "${RED}⚠️  AG06 not detected - continuing anyway${NC}"
fi

# 4. Test AI modules
echo -e "${YELLOW}Step 4: Testing AI modules...${NC}"
python3 -c "
from ai_mixing_brain import AutonomousMixingEngine
from studio_dsp_chain import StudioDSPChain
print('✅ AI Mixing Engine loaded')
print('✅ Studio DSP Chain loaded')
"

# 5. Launch with Gunicorn
echo -e "${BLUE}Step 5: Starting AI Mixing Studio...${NC}"
echo "Configuration:"
echo "  - Port: ${API_PORT}"
echo "  - Workers: ${WORKERS}"
echo "  - Worker class: ${WORKER_CLASS}"
echo "  - Audio: ${AUDIO_RATE}Hz, ${AUDIO_BLOCK} samples"
echo ""
echo -e "${GREEN}🎵 AI Mixing Features:${NC}"
echo "  - Autonomous genre detection"
echo "  - Adaptive DSP processing"
echo "  - Studio-quality effects chain"
echo "  - Real-time streaming dashboard"
echo ""
echo -e "${MAGENTA}Dashboard: http://localhost:${API_PORT}${NC}"
echo ""

# Launch Gunicorn with AI app
exec gunicorn \
    --workers ${WORKERS} \
    --worker-class ${WORKER_CLASS} \
    --worker-connections ${WORKER_CONNECTIONS} \
    --timeout ${TIMEOUT} \
    --bind ${API_HOST}:${API_PORT} \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL} \
    --preload \
    production_app_ai:app