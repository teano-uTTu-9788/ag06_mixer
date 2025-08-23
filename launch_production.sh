#!/bin/bash
# Production launch script for AG06 mixer
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
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 AG06 Production Server Launch${NC}"
echo "=================================="

# 1. Run port guard
echo -e "${YELLOW}Step 1: Port guard...${NC}"
./port_guard.sh

# 2. Health check dependencies
echo -e "${YELLOW}Step 2: Checking dependencies...${NC}"
python3 -c "import flask, pyaudio, numpy, gunicorn, gevent" || {
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

# 4. Launch with Gunicorn
echo -e "${YELLOW}Step 4: Starting Gunicorn...${NC}"
echo "Configuration:"
echo "  - Port: ${API_PORT}"
echo "  - Workers: ${WORKERS}"
echo "  - Worker class: ${WORKER_CLASS}"
echo "  - Audio: ${AUDIO_RATE}Hz, ${AUDIO_BLOCK} samples"

# Launch Gunicorn
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
    production_app:app