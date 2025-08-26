#!/usr/bin/env python3
"""
Interactive Real-Time Dashboard System - Phase 2
Advanced visualization and monitoring interface with WebSocket real-time updates
"""

import asyncio
import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Web framework and real-time capabilities
try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask/SocketIO not available - using HTTP-only fallback")

# Import our existing systems
from integrated_workflow_system import IntegratedWorkflowSystem
from performance_optimization_monitoring import PerformanceOptimizationMonitor
from ml_predictive_analytics_engine import MLPredictiveAnalyticsEngine

class DashboardComponent(Enum):
    SYSTEM_OVERVIEW = "system_overview"
    PERFORMANCE_METRICS = "performance_metrics"
    ML_INSIGHTS = "ml_insights"
    WORKFLOW_STATUS = "workflow_status"
    ALERT_CENTER = "alert_center"
    RESOURCE_USAGE = "resource_usage"
    TREND_ANALYSIS = "trend_analysis"
    PREDICTIVE_ALERTS = "predictive_alerts"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"

@dataclass
class DashboardAlert:
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime
    auto_dismiss: bool = False
    dismiss_after_seconds: int = 30

@dataclass
class MetricTile:
    tile_id: str
    title: str
    value: str
    change_percent: float
    trend: str  # up, down, stable
    status: str  # good, warning, critical
    sparkline_data: List[float]
    last_updated: datetime

class InteractiveDashboardSystem:
    """Advanced interactive dashboard with real-time updates"""
    
    def __init__(self, dashboard_id: str = "dashboard_001", port: int = 8081):
        self.dashboard_id = dashboard_id
        self.port = port
        self.app = None
        self.socketio = None
        
        # System integrations
        self.workflow_system = None
        self.performance_monitor = None
        self.ml_engine = None
        
        # Dashboard state
        self.active_connections = set()
        self.dashboard_data = {}
        self.alerts = []
        self.metric_tiles = {}
        self.update_intervals = {
            DashboardComponent.SYSTEM_OVERVIEW: 5,      # 5 seconds
            DashboardComponent.PERFORMANCE_METRICS: 10,  # 10 seconds
            DashboardComponent.ML_INSIGHTS: 60,         # 1 minute
            DashboardComponent.WORKFLOW_STATUS: 15,     # 15 seconds
            DashboardComponent.ALERT_CENTER: 5,         # 5 seconds
        }
        
        # Configuration
        self.config = {
            "max_alerts": 100,
            "max_sparkline_points": 50,
            "auto_refresh_enabled": True,
            "theme": "dark",
            "show_debug_info": False
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        print(f"📊 Interactive Dashboard System {self.dashboard_id} initialized")
        print(f"   ✅ Port: {self.port}")
        print(f"   ✅ WebSocket support: {'Yes' if FLASK_AVAILABLE else 'No (HTTP fallback)'}")
        print(f"   ✅ Components: {len(DashboardComponent)} available")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the dashboard"""
        logger = logging.getLogger(f"dashboard_{self.dashboard_id}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s | DASHBOARD-{self.dashboard_id} | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the dashboard and all integrations"""
        try:
            self.logger.info("📊 Initializing Interactive Dashboard System...")
            
            # Initialize system integrations
            self.workflow_system = IntegratedWorkflowSystem()
            
            self.performance_monitor = PerformanceOptimizationMonitor()
            await self.performance_monitor.initialize()
            
            self.ml_engine = MLPredictiveAnalyticsEngine()
            await self.ml_engine.initialize()
            
            # Initialize Flask app if available
            if FLASK_AVAILABLE:
                await self._initialize_flask_app()
            
            # Initialize dashboard data
            await self._initialize_dashboard_data()
            
            self.logger.info("✅ Interactive Dashboard System fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def _initialize_flask_app(self):
        """Initialize Flask application with SocketIO"""
        if not FLASK_AVAILABLE:
            return
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'dashboard_secret_key_2025'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        @self.app.route('/')
        async def index():
            return self._generate_dashboard_html()
        
        @self.app.route('/api/data')
        async def get_dashboard_data():
            return jsonify(await self._get_current_dashboard_data())
        
        @self.app.route('/api/alerts')
        async def get_alerts():
            return jsonify([asdict(alert) for alert in self.alerts])
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        async def dashboard_config():
            if request.method == 'POST':
                new_config = request.json
                self.config.update(new_config)
                return jsonify({"status": "updated", "config": self.config})
            return jsonify(self.config)
        
        # SocketIO events
        @self.socketio.on('connect')
        def handle_connect():
            self.active_connections.add(request.sid)
            self.logger.info(f"Client connected: {request.sid}")
            emit('connection_established', {'dashboard_id': self.dashboard_id})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.active_connections.discard(request.sid)
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            component = data.get('component', 'all')
            # Trigger immediate update for requested component
            self.socketio.start_background_task(self._send_component_update, component)
    
    async def _initialize_dashboard_data(self):
        """Initialize dashboard with initial data"""
        
        # Initialize metric tiles
        self.metric_tiles = {
            "system_health": MetricTile(
                tile_id="system_health",
                title="System Health",
                value="100%",
                change_percent=0.0,
                trend="stable",
                status="good",
                sparkline_data=[100.0] * 10,
                last_updated=datetime.now()
            ),
            "active_workflows": MetricTile(
                tile_id="active_workflows",
                title="Active Workflows",
                value="0",
                change_percent=0.0,
                trend="stable",
                status="good",
                sparkline_data=[0.0] * 10,
                last_updated=datetime.now()
            ),
            "cpu_usage": MetricTile(
                tile_id="cpu_usage",
                title="CPU Usage",
                value="0%",
                change_percent=0.0,
                trend="stable",
                status="good",
                sparkline_data=[0.0] * 10,
                last_updated=datetime.now()
            ),
            "memory_usage": MetricTile(
                tile_id="memory_usage",
                title="Memory Usage",
                value="0%",
                change_percent=0.0,
                trend="stable",
                status="good",
                sparkline_data=[0.0] * 10,
                last_updated=datetime.now()
            )
        }
        
        # Add welcome alert
        self.add_alert(
            AlertSeverity.SUCCESS,
            "Dashboard Initialized",
            "Interactive dashboard system is now online and collecting data",
            "system",
            auto_dismiss=True,
            dismiss_after_seconds=10
        )
    
    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML interface"""
        
        dashboard_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AG06 Production Dashboard</title>
            <style>
                :root {
                    --primary-color: #2c3e50;
                    --secondary-color: #3498db;
                    --success-color: #27ae60;
                    --warning-color: #f39c12;
                    --danger-color: #e74c3c;
                    --dark-bg: #1a1a1a;
                    --card-bg: #2d2d2d;
                    --text-primary: #ffffff;
                    --text-secondary: #bdc3c7;
                }
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: var(--dark-bg);
                    color: var(--text-primary);
                    line-height: 1.6;
                }
                
                .dashboard-header {
                    background: var(--primary-color);
                    padding: 1rem 2rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                }
                
                .dashboard-title {
                    font-size: 1.8rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .status-indicator {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: var(--success-color);
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                .dashboard-container {
                    padding: 2rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1.5rem;
                }
                
                .metric-card {
                    background: var(--card-bg);
                    border-radius: 12px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
                    transition: transform 0.3s ease;
                }
                
                .metric-card:hover {
                    transform: translateY(-2px);
                }
                
                .metric-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                }
                
                .metric-title {
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                .metric-value {
                    font-size: 2.2rem;
                    font-weight: 700;
                    color: var(--text-primary);
                }
                
                .metric-change {
                    font-size: 0.8rem;
                    padding: 0.2rem 0.5rem;
                    border-radius: 4px;
                    margin-top: 0.5rem;
                }
                
                .metric-change.positive {
                    background: rgba(39, 174, 96, 0.2);
                    color: var(--success-color);
                }
                
                .metric-change.negative {
                    background: rgba(231, 76, 60, 0.2);
                    color: var(--danger-color);
                }
                
                .metric-change.neutral {
                    background: rgba(149, 165, 166, 0.2);
                    color: var(--text-secondary);
                }
                
                .sparkline {
                    height: 40px;
                    background: rgba(52, 152, 219, 0.1);
                    border-radius: 4px;
                    margin-top: 1rem;
                    position: relative;
                    overflow: hidden;
                }
                
                .alert-center {
                    grid-column: 1 / -1;
                    background: var(--card-bg);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                }
                
                .alert-item {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding: 0.75rem 1rem;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                    transition: all 0.3s ease;
                }
                
                .alert-item.info {
                    background: rgba(52, 152, 219, 0.1);
                    border-left: 4px solid var(--secondary-color);
                }
                
                .alert-item.success {
                    background: rgba(39, 174, 96, 0.1);
                    border-left: 4px solid var(--success-color);
                }
                
                .alert-item.warning {
                    background: rgba(243, 156, 18, 0.1);
                    border-left: 4px solid var(--warning-color);
                }
                
                .alert-item.critical {
                    background: rgba(231, 76, 60, 0.1);
                    border-left: 4px solid var(--danger-color);
                }
                
                .alert-icon {
                    font-size: 1.2rem;
                }
                
                .alert-content {
                    flex: 1;
                }
                
                .alert-title {
                    font-weight: 600;
                    margin-bottom: 0.2rem;
                }
                
                .alert-message {
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                }
                
                .alert-timestamp {
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                    opacity: 0.7;
                }
                
                .connection-status {
                    position: fixed;
                    top: 1rem;
                    right: 1rem;
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    z-index: 1000;
                }
                
                .connection-status.connected {
                    background: var(--success-color);
                    color: white;
                }
                
                .connection-status.disconnected {
                    background: var(--danger-color);
                    color: white;
                }
                
                .loading-spinner {
                    border: 2px solid rgba(255,255,255,0.1);
                    border-radius: 50%;
                    border-top: 2px solid var(--secondary-color);
                    width: 20px;
                    height: 20px;
                    animation: spin 1s linear infinite;
                    display: inline-block;
                    margin-left: 0.5rem;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .dashboard-footer {
                    text-align: center;
                    padding: 2rem;
                    color: var(--text-secondary);
                    font-size: 0.9rem;
                }
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        </head>
        <body>
            <div class="dashboard-header">
                <div class="dashboard-title">
                    <span>🎛️ AG06 Production Dashboard</span>
                    <div class="status-indicator" id="statusIndicator"></div>
                    <span style="font-size: 0.9rem; margin-left: auto;" id="lastUpdate">Initializing...</span>
                </div>
            </div>
            
            <div class="connection-status connected" id="connectionStatus">
                ● Connected
            </div>
            
            <div class="dashboard-container">
                <div class="alert-center">
                    <h3 style="margin-bottom: 1rem;">📢 System Alerts</h3>
                    <div id="alertContainer">
                        <div class="alert-item info">
                            <div class="alert-icon">ℹ️</div>
                            <div class="alert-content">
                                <div class="alert-title">Dashboard Loading</div>
                                <div class="alert-message">Connecting to real-time data feeds...</div>
                            </div>
                            <div class="alert-timestamp">Just now</div>
                        </div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">System Health</div>
                        <div class="loading-spinner" id="healthSpinner"></div>
                    </div>
                    <div class="metric-value" id="systemHealth">--</div>
                    <div class="metric-change neutral" id="healthChange">Initializing</div>
                    <div class="sparkline" id="healthSparkline"></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Active Workflows</div>
                    </div>
                    <div class="metric-value" id="activeWorkflows">--</div>
                    <div class="metric-change neutral" id="workflowChange">Loading...</div>
                    <div class="sparkline" id="workflowSparkline"></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">CPU Usage</div>
                    </div>
                    <div class="metric-value" id="cpuUsage">--%</div>
                    <div class="metric-change neutral" id="cpuChange">Monitoring...</div>
                    <div class="sparkline" id="cpuSparkline"></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Memory Usage</div>
                    </div>
                    <div class="metric-value" id="memoryUsage">--%</div>
                    <div class="metric-change neutral" id="memoryChange">Collecting...</div>
                    <div class="sparkline" id="memorySparkline"></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">ML Predictions</div>
                    </div>
                    <div class="metric-value" id="mlPredictions">--</div>
                    <div class="metric-change neutral" id="mlChange">Analyzing...</div>
                    <div class="sparkline" id="mlSparkline"></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Response Time</div>
                    </div>
                    <div class="metric-value" id="responseTime">--ms</div>
                    <div class="metric-change neutral" id="responseChange">Testing...</div>
                    <div class="sparkline" id="responseSparkline"></div>
                </div>
            </div>
            
            <div class="dashboard-footer">
                🏭 AG06 Production Dashboard - Real-time monitoring with ML-powered insights<br>
                Last system update: <span id="footerTimestamp">--</span>
            </div>
            
            <script>
                // Dashboard JavaScript
                let socket;
                let dashboardData = {};
                let isConnected = false;
                
                // Initialize WebSocket connection
                function initializeSocket() {
                    if (typeof io !== 'undefined') {
                        socket = io();
                        
                        socket.on('connect', function() {
                            isConnected = true;
                            updateConnectionStatus();
                            console.log('Connected to dashboard server');
                        });
                        
                        socket.on('disconnect', function() {
                            isConnected = false;
                            updateConnectionStatus();
                            console.log('Disconnected from dashboard server');
                        });
                        
                        socket.on('dashboard_update', function(data) {
                            updateDashboard(data);
                        });
                        
                        socket.on('new_alert', function(alert) {
                            addAlert(alert);
                        });
                        
                    } else {
                        // Fallback to HTTP polling
                        setInterval(fetchDashboardData, 10000);
                        console.log('Using HTTP polling fallback');
                    }
                }
                
                // Update connection status
                function updateConnectionStatus() {
                    const statusElement = document.getElementById('connectionStatus');
                    const indicatorElement = document.getElementById('statusIndicator');
                    
                    if (isConnected) {
                        statusElement.className = 'connection-status connected';
                        statusElement.textContent = '● Connected';
                        indicatorElement.style.background = 'var(--success-color)';
                    } else {
                        statusElement.className = 'connection-status disconnected';
                        statusElement.textContent = '● Disconnected';
                        indicatorElement.style.background = 'var(--danger-color)';
                    }
                }
                
                // Fetch dashboard data (HTTP fallback)
                async function fetchDashboardData() {
                    try {
                        const response = await fetch('/api/data');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Failed to fetch dashboard data:', error);
                    }
                }
                
                // Update dashboard with new data
                function updateDashboard(data) {
                    dashboardData = data;
                    
                    // Update metrics
                    if (data.metrics) {
                        updateMetric('systemHealth', data.metrics.system_health || '100%');
                        updateMetric('activeWorkflows', data.metrics.active_workflows || '0');
                        updateMetric('cpuUsage', data.metrics.cpu_usage || '0%');
                        updateMetric('memoryUsage', data.metrics.memory_usage || '0%');
                        updateMetric('mlPredictions', data.metrics.ml_predictions || '0');
                        updateMetric('responseTime', data.metrics.response_time || '0ms');
                    }
                    
                    // Update timestamp
                    const now = new Date().toLocaleTimeString();
                    document.getElementById('lastUpdate').textContent = `Last updated: ${now}`;
                    document.getElementById('footerTimestamp').textContent = now;
                    
                    // Remove loading spinners
                    document.querySelectorAll('.loading-spinner').forEach(spinner => {
                        spinner.style.display = 'none';
                    });
                }
                
                // Update individual metric
                function updateMetric(metricId, value, change = null, trend = 'neutral') {
                    const valueElement = document.getElementById(metricId);
                    const changeElement = document.getElementById(metricId.replace(/([A-Z])/g, '$1').toLowerCase() + 'Change');
                    
                    if (valueElement) {
                        valueElement.textContent = value;
                    }
                    
                    if (changeElement && change !== null) {
                        changeElement.textContent = change;
                        changeElement.className = `metric-change ${trend}`;
                    }
                }
                
                // Add new alert
                function addAlert(alert) {
                    const container = document.getElementById('alertContainer');
                    const alertElement = document.createElement('div');
                    
                    alertElement.className = `alert-item ${alert.severity}`;
                    alertElement.innerHTML = `
                        <div class="alert-icon">${getAlertIcon(alert.severity)}</div>
                        <div class="alert-content">
                            <div class="alert-title">${alert.title}</div>
                            <div class="alert-message">${alert.message}</div>
                        </div>
                        <div class="alert-timestamp">${formatTimestamp(alert.timestamp)}</div>
                    `;
                    
                    container.insertBefore(alertElement, container.firstChild);
                    
                    // Auto-dismiss if configured
                    if (alert.auto_dismiss) {
                        setTimeout(() => {
                            alertElement.remove();
                        }, (alert.dismiss_after_seconds || 30) * 1000);
                    }
                    
                    // Limit number of alerts
                    const alerts = container.querySelectorAll('.alert-item');
                    if (alerts.length > 10) {
                        alerts[alerts.length - 1].remove();
                    }
                }
                
                // Get alert icon
                function getAlertIcon(severity) {
                    switch (severity) {
                        case 'success': return '✅';
                        case 'warning': return '⚠️';
                        case 'critical': return '🚨';
                        default: return 'ℹ️';
                    }
                }
                
                // Format timestamp
                function formatTimestamp(timestamp) {
                    const date = new Date(timestamp);
                    return date.toLocaleTimeString();
                }
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    initializeSocket();
                    
                    // Initial data fetch
                    setTimeout(fetchDashboardData, 1000);
                    
                    // Set up periodic updates if no WebSocket
                    if (typeof io === 'undefined') {
                        setInterval(fetchDashboardData, 10000);
                    }
                });
            </script>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def add_alert(self, severity: AlertSeverity, title: str, message: str, 
                  component: str, auto_dismiss: bool = False, 
                  dismiss_after_seconds: int = 30):
        """Add a new alert to the dashboard"""
        
        alert = DashboardAlert(
            alert_id=f"alert_{int(time.time())}_{len(self.alerts)}",
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=datetime.now(),
            auto_dismiss=auto_dismiss,
            dismiss_after_seconds=dismiss_after_seconds
        )
        
        self.alerts.append(alert)
        
        # Keep alerts manageable
        if len(self.alerts) > self.config["max_alerts"]:
            self.alerts = self.alerts[-self.config["max_alerts"]:]
        
        # Send real-time update if WebSocket available
        if FLASK_AVAILABLE and self.socketio and self.active_connections:
            self.socketio.emit('new_alert', asdict(alert), room=None)
        
        self.logger.info(f"📢 Alert added: {severity.value.upper()} - {title}")
    
    async def update_metric_tile(self, tile_id: str, value: str, 
                               change_percent: float = None,
                               sparkline_point: float = None):
        """Update a metric tile with new data"""
        
        if tile_id not in self.metric_tiles:
            return
        
        tile = self.metric_tiles[tile_id]
        tile.value = value
        tile.last_updated = datetime.now()
        
        if change_percent is not None:
            tile.change_percent = change_percent
            tile.trend = "up" if change_percent > 0 else "down" if change_percent < 0 else "stable"
            
            # Update status based on the metric type and change
            if tile_id in ["cpu_usage", "memory_usage"]:
                current_val = float(value.replace('%', '').replace('ms', ''))
                if current_val > 90:
                    tile.status = "critical"
                elif current_val > 80:
                    tile.status = "warning"
                else:
                    tile.status = "good"
        
        if sparkline_point is not None:
            tile.sparkline_data.append(sparkline_point)
            if len(tile.sparkline_data) > self.config["max_sparkline_points"]:
                tile.sparkline_data = tile.sparkline_data[-self.config["max_sparkline_points"]:]
        
        self.metric_tiles[tile_id] = tile
    
    async def _get_current_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for API/WebSocket"""
        
        try:
            # Collect system metrics
            metrics = {}
            
            # Get workflow system data
            if self.workflow_system:
                try:
                    health = await self.workflow_system.get_system_health()
                    metrics["system_health"] = "Healthy" if health.get("overall_status") == "healthy" else "Warning"
                    metrics["active_workflows"] = str(health.get("active_workflows", 0))
                except Exception as e:
                    self.logger.warning(f"Failed to get workflow metrics: {e}")
                    metrics["system_health"] = "Unknown"
                    metrics["active_workflows"] = "0"
            
            # Get performance data
            if self.performance_monitor:
                try:
                    # Use a mock performance data for now since the method doesn't exist
                    import psutil
                    metrics["cpu_usage"] = f"{psutil.cpu_percent()}%"
                    metrics["memory_usage"] = f"{psutil.virtual_memory().percent}%"
                    
                    # Update metric tiles
                    await self.update_metric_tile("cpu_usage", metrics["cpu_usage"], 
                                                sparkline_point=psutil.cpu_percent())
                    await self.update_metric_tile("memory_usage", metrics["memory_usage"],
                                                sparkline_point=psutil.virtual_memory().percent)
                except Exception as e:
                    self.logger.warning(f"Failed to get performance metrics: {e}")
                    metrics["cpu_usage"] = "0%"
                    metrics["memory_usage"] = "0%"
            
            # Get ML insights
            if self.ml_engine:
                try:
                    status = await self.ml_engine.get_analytics_status()
                    metrics["ml_predictions"] = str(status.get("metrics", {}).get("total_predictions", 0))
                except Exception as e:
                    self.logger.warning(f"Failed to get ML metrics: {e}")
                    metrics["ml_predictions"] = "0"
            
            # Response time (mock data)
            metrics["response_time"] = "124ms"
            
            return {
                "metrics": metrics,
                "metric_tiles": {k: asdict(v) for k, v in self.metric_tiles.items()},
                "alerts": [asdict(alert) for alert in self.alerts[-10:]],  # Last 10 alerts
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def _send_component_update(self, component: str = "all"):
        """Send real-time update for specific component"""
        if not FLASK_AVAILABLE or not self.socketio or not self.active_connections:
            return
        
        try:
            data = await self._get_current_dashboard_data()
            self.socketio.emit('dashboard_update', data, room=None)
        except Exception as e:
            self.logger.error(f"Error sending component update: {e}")
    
    async def start_real_time_updates(self):
        """Start background task for real-time dashboard updates"""
        
        self.logger.info("🔄 Starting real-time dashboard updates...")
        
        async def update_loop():
            while True:
                try:
                    # Send updates to connected clients
                    if self.active_connections:
                        await self._send_component_update()
                    
                    # Check for alerts based on metrics
                    await self._check_alert_conditions()
                    
                    await asyncio.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    await asyncio.sleep(30)  # Wait longer on error
        
        # Start the update loop as a background task
        asyncio.create_task(update_loop())
    
    async def _check_alert_conditions(self):
        """Check system conditions and generate alerts"""
        
        try:
            data = await self._get_current_dashboard_data()
            metrics = data.get("metrics", {})
            
            # Check CPU usage
            cpu_usage = float(metrics.get("cpu_usage", "0%").replace("%", ""))
            if cpu_usage > 90:
                self.add_alert(
                    AlertSeverity.CRITICAL,
                    "High CPU Usage",
                    f"CPU usage is at {cpu_usage}% - immediate attention required",
                    "performance",
                    auto_dismiss=True,
                    dismiss_after_seconds=60
                )
            elif cpu_usage > 80:
                self.add_alert(
                    AlertSeverity.WARNING,
                    "Elevated CPU Usage", 
                    f"CPU usage is at {cpu_usage}% - monitor closely",
                    "performance",
                    auto_dismiss=True,
                    dismiss_after_seconds=45
                )
            
            # Check memory usage
            memory_usage = float(metrics.get("memory_usage", "0%").replace("%", ""))
            if memory_usage > 90:
                self.add_alert(
                    AlertSeverity.CRITICAL,
                    "High Memory Usage",
                    f"Memory usage is at {memory_usage}% - system may become unstable",
                    "performance",
                    auto_dismiss=True,
                    dismiss_after_seconds=60
                )
            
        except Exception as e:
            self.logger.warning(f"Error checking alert conditions: {e}")
    
    async def run_dashboard_server(self):
        """Run the dashboard server"""
        
        if not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - dashboard server cannot start")
            return
        
        try:
            self.logger.info(f"🚀 Starting dashboard server on port {self.port}...")
            
            # Start real-time updates
            await self.start_real_time_updates()
            
            # Run the Flask-SocketIO server
            self.socketio.run(
                self.app,
                host='0.0.0.0',
                port=self.port,
                debug=False,
                allow_unsafe_werkzeug=True
            )
            
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")
            self.logger.error(traceback.format_exc())
    
    async def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard system status"""
        
        return {
            "dashboard_id": self.dashboard_id,
            "status": "operational" if self.app else "not_started",
            "port": self.port,
            "active_connections": len(self.active_connections),
            "total_alerts": len(self.alerts),
            "metric_tiles": len(self.metric_tiles),
            "websocket_support": FLASK_AVAILABLE,
            "config": self.config,
            "uptime_seconds": (datetime.now() - datetime.now()).total_seconds()
        }
    
    async def run_dashboard_demo(self) -> Dict[str, Any]:
        """Run dashboard demonstration"""
        
        self.logger.info("🎯 Starting dashboard demonstration...")
        
        start_time = datetime.now()
        
        # Add some demo alerts
        demo_alerts = [
            (AlertSeverity.SUCCESS, "System Online", "Dashboard system successfully initialized", "system"),
            (AlertSeverity.INFO, "Data Collection", "Starting real-time metrics collection", "monitoring"),
            (AlertSeverity.WARNING, "Resource Monitor", "CPU usage trending upward", "performance"),
        ]
        
        for severity, title, message, component in demo_alerts:
            self.add_alert(severity, title, message, component, auto_dismiss=True)
            await asyncio.sleep(0.5)
        
        # Update some metric tiles with demo data
        demo_metrics = [
            ("system_health", "98.5%", 0.2, 98.5),
            ("active_workflows", "7", 16.7, 7.0),
            ("cpu_usage", "45%", 12.5, 45.0),
            ("memory_usage", "67%", -3.2, 67.0),
        ]
        
        for tile_id, value, change, sparkline in demo_metrics:
            await self.update_metric_tile(tile_id, value, change, sparkline)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        demo_results = {
            "demo_summary": {
                "processing_time_seconds": processing_time,
                "alerts_created": len(demo_alerts),
                "metric_tiles_updated": len(demo_metrics),
                "dashboard_url": f"http://localhost:{self.port}"
            },
            "dashboard_status": await self.get_dashboard_status(),
            "current_data": await self._get_current_dashboard_data()
        }
        
        self.logger.info(f"🎉 Dashboard demo complete - {len(demo_alerts)} alerts, {len(demo_metrics)} metrics updated")
        
        return demo_results

async def main():
    """Main entry point for Interactive Dashboard System"""
    print("📊 Starting Interactive Dashboard System - Phase 2")
    print("=" * 80)
    
    # Initialize dashboard
    dashboard = InteractiveDashboardSystem()
    
    if not await dashboard.initialize():
        print("❌ Failed to initialize dashboard system")
        return
    
    # Run demonstration
    demo_results = await dashboard.run_dashboard_demo()
    
    print("\n" + "=" * 80)
    print("📋 Dashboard Demo Results:")
    print(f"   Alerts Created: {demo_results['demo_summary']['alerts_created']}")
    print(f"   Metric Tiles Updated: {demo_results['demo_summary']['metric_tiles_updated']}")
    print(f"   Processing Time: {demo_results['demo_summary']['processing_time_seconds']:.2f}s")
    print(f"   Dashboard URL: {demo_results['demo_summary']['dashboard_url']}")
    print(f"   WebSocket Support: {'Yes' if FLASK_AVAILABLE else 'No (HTTP fallback)'}")
    
    # Export results
    results_file = "interactive_dashboard_results.json"
    with open(results_file, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📄 Full results exported: {results_file}")
    
    if FLASK_AVAILABLE:
        print(f"\n🌐 Dashboard available at: http://localhost:{dashboard.port}")
        print("   Press Ctrl+C to stop the dashboard server")
        
        try:
            # Start the dashboard server
            await dashboard.run_dashboard_server()
        except KeyboardInterrupt:
            print("\n⚠️  Dashboard server stopped")
    else:
        print("\n⚠️  Flask not available - install Flask and Flask-SocketIO for full functionality")
    
    print("\n✅ Interactive Dashboard System Phase 2 demonstration complete!")

if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import flask
        import flask_socketio
    except ImportError as e:
        print(f"⚠️  Missing web dependency: {e}")
        print("Installing required packages...")
        import subprocess
        packages = ["flask", "flask-socketio", "psutil"]
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except:
                print(f"❌ Failed to install {package} - continuing with fallbacks")
    
    asyncio.run(main())