#!/usr/bin/env python3
"""
Simple server that just works - no fancy features
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

webapp_dir = '/Users/nguythe/ag06_mixer/automation-framework/webapp'

@app.route('/')
def index():
    return send_from_directory(webapp_dir, 'ai_mixer.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory(webapp_dir, path)

# Minimal API endpoints for the webapp
@app.route('/api/status')
def status():
    return jsonify({
        'processing': False,
        'hardware': {'ag06_connected': True, 'device_id': 1},
        'input_level': {'rms': -30, 'peak': -25, 'clipping': False, 'too_quiet': False},
        'spectrum': [10] * 64,
        'music': {'detected': False, 'confidence': 0},
        'voice': {'detected': False, 'confidence': 0},
        'timestamp': 0
    })

@app.route('/api/start', methods=['POST'])
def start():
    return jsonify({'success': True, 'status': 'started', 'message': 'Started'})

@app.route('/api/stop', methods=['POST'])
def stop():
    return jsonify({'success': True, 'status': 'stopped', 'message': 'Stopped'})

@app.route('/api/spectrum')
def spectrum():
    return jsonify({
        'spectrum': [20] * 64,
        'level_db': -30,
        'peak_db': -25,
        'classification': 'music',
        'peak_frequency': 440,
        'timestamp': 0
    })

if __name__ == '__main__':
    print("Simple server starting on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)