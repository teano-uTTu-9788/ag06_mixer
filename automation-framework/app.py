from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add .aican/agents to path to import AG06 agent
# Assuming structure:
# automation-framework/
#   app.py
#   .aican/
#     agents/
#       ag06_integration_agent.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.aican', 'agents'))

try:
    from ag06_integration_agent import AG06IntegrationAgent
    agent_available = True
except ImportError as e:
    logger.error(f"Failed to import AG06 Agent: {e}")
    agent_available = False

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app connectivity

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for mobile app connection testing."""
    return jsonify({
        "status": "ok",
        "component": "AG06 Backend",
        "agent_available": agent_available,
        "version": "1.1.0 (Audio Enabled)"
    })

@app.route('/api/devices', methods=['GET'])
def list_devices():
    """List detected AG06 audio devices."""
    if not agent_available:
         return jsonify({"error": "Agent not available", "devices": []}), 503
    
    try:
        # Instantiate agent on demand or use global if preferred (creating new for statelessness mostly)
        # Note: In a real app we'd want a singleton, but this is fine for MVP verification
        agent = AG06IntegrationAgent()
        
        # detect_ag06_device is async, so we run it synchronously here
        devices = asyncio.run(agent.detect_ag06_device())
        
        # If no AG06 found, let's also just query sounddevice directly to show *something*
        # so the user knows the server is actually checking hardware
        if not devices:
            import sounddevice as sd
            all_devices = sd.query_devices()
            # Convert QueryList to list of dicts for JSON serialization
            simple_device_list = []
            for d in all_devices:
                simple_device_list.append({
                    "name": d['name'],
                    "channels": d['max_input_channels']
                })
            
            return jsonify({
                "status": "ok", 
                "message": "No AG06 detected. Listing all server devices.",
                "ag06_found": False,
                "all_devices": simple_device_list
            })

        return jsonify({"status": "ok", "ag06_found": True, "devices": devices})
    except Exception as e:
        logger.error(f"Device scan failed: {e}")
        return jsonify({"status": "error", "message": str(e), "devices": []})

@app.route('/api/monitor/start', methods=['POST'])
def start_monitor():
    """Trigger a 10s audio monitoring test."""
    if not agent_available:
         return jsonify({"error": "Agent not available"}), 503
    
    def run_test():
        agent = AG06IntegrationAgent()
        # Mock device detection if needed for the test to run without physical AG06
        # modifying agent to be more permissive might be needed, but let's try standard first
        asyncio.run(agent.test_real_time_audio())
    
    import threading
    thread = threading.Thread(target=run_test)
    thread.start()
    
    return jsonify({"status": "started", "message": "Audio monitoring test initiated (10s log)"})

@app.route('/api/process', methods=['POST'])
def process_audio():
    """Stub endpoint for audio processing."""
    if not agent_available:
        return jsonify({"error": "AG06 Agent not available"}), 503
    
    return jsonify({
        "status": "processing_started",
        "message": "Audio processing request received (Stub)"
    })

if __name__ == '__main__':
    # Listen on all interfaces for mobile emulator access
    app.run(host='0.0.0.0', port=8899)
