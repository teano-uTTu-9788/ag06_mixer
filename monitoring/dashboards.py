"""
Monitoring Dashboards for Production
MANU Compliance: Observability Requirements
"""
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricData:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str]
    metric_name: str


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str  # line_chart, gauge, counter, table
    title: str
    metric_queries: List[str]
    refresh_interval: int = 30  # seconds
    config: Dict[str, Any] = None


class MetricsCollector:
    """
    Collects metrics from various system components
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics_buffer = []
        self.max_buffer_size = 10000
        self.collectors = {}
        
    def add_collector(self, name: str, collector_func):
        """
        Add metric collector function
        
        Args:
            name: Collector name
            collector_func: Function that returns metrics
        """
        self.collectors[name] = collector_func
    
    async def collect_all_metrics(self) -> List[MetricData]:
        """
        Collect metrics from all registered collectors
        
        Returns:
            List of metric data points
        """
        all_metrics = []
        current_time = time.time()
        
        for name, collector in self.collectors.items():
            try:
                metrics = await self._safe_collect(collector)
                for metric in metrics:
                    metric.timestamp = current_time
                    metric.tags['collector'] = name
                all_metrics.append(metric)
            except Exception as e:
                # Log error but continue collecting other metrics
                error_metric = MetricData(
                    timestamp=current_time,
                    value=1.0,
                    tags={'collector': name, 'error': str(e)},
                    metric_name='collector_error'
                )
                all_metrics.append(error_metric)
        
        return all_metrics
    
    async def _safe_collect(self, collector_func) -> List[MetricData]:
        """Safely execute collector function"""
        try:
            if asyncio.iscoroutinefunction(collector_func):
                return await collector_func()
            else:
                return collector_func()
        except Exception as e:
            return [MetricData(
                timestamp=time.time(),
                value=0.0,
                tags={'error': str(e)},
                metric_name='collection_failed'
            )]
    
    def store_metrics(self, metrics: List[MetricData]):
        """Store metrics in buffer"""
        self.metrics_buffer.extend(metrics)
        
        # Keep buffer size manageable
        if len(self.metrics_buffer) > self.max_buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size:]
    
    def query_metrics(self, 
                     metric_name: str,
                     time_range: int = 3600,  # seconds
                     tags: Optional[Dict[str, str]] = None) -> List[MetricData]:
        """
        Query metrics from buffer
        
        Args:
            metric_name: Name of metric
            time_range: Time range in seconds from now
            tags: Tag filters
            
        Returns:
            Filtered metrics
        """
        cutoff_time = time.time() - time_range
        
        filtered_metrics = []
        for metric in self.metrics_buffer:
            # Time filter
            if metric.timestamp < cutoff_time:
                continue
                
            # Metric name filter
            if metric.metric_name != metric_name:
                continue
                
            # Tag filters
            if tags:
                match = True
                for key, value in tags.items():
                    if key not in metric.tags or metric.tags[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            filtered_metrics.append(metric)
        
        return sorted(filtered_metrics, key=lambda m: m.timestamp)


class Dashboard:
    """
    Production monitoring dashboard
    """
    
    def __init__(self, dashboard_id: str, title: str):
        """
        Initialize dashboard
        
        Args:
            dashboard_id: Dashboard identifier
            title: Dashboard title
        """
        self.dashboard_id = dashboard_id
        self.title = title
        self.widgets = []
        self.metrics_collector = MetricsCollector()
        self.is_running = False
        
        # Setup default collectors
        self._setup_default_collectors()
    
    def _setup_default_collectors(self):
        """Setup default system metric collectors"""
        
        def system_metrics():
            """Collect system metrics"""
            import psutil
            
            return [
                MetricData(
                    timestamp=time.time(),
                    value=psutil.cpu_percent(),
                    tags={'component': 'system'},
                    metric_name='cpu_percent'
                ),
                MetricData(
                    timestamp=time.time(),
                    value=psutil.virtual_memory().percent,
                    tags={'component': 'system'},
                    metric_name='memory_percent'
                ),
                MetricData(
                    timestamp=time.time(),
                    value=psutil.disk_usage('/').percent,
                    tags={'component': 'system'},
                    metric_name='disk_percent'
                )
            ]
        
        def audio_metrics():
            """Collect AG-06 specific metrics"""
            # Read from optimization status
            try:
                status_file = Path('/Users/nguythe/ag06_mixer/ag06_optimization_status.json')
                if status_file.exists():
                    with open(status_file) as f:
                        data = json.load(f)
                    
                    return [
                        MetricData(
                            timestamp=time.time(),
                            value=float(data.get('optimizations', 0)),
                            tags={'component': 'ag06'},
                            metric_name='total_optimizations'
                        ),
                        MetricData(
                            timestamp=time.time(),
                            value=1.0 if data.get('running', False) else 0.0,
                            tags={'component': 'ag06'},
                            metric_name='agent_status'
                        )
                    ]
                else:
                    return []
            except Exception:
                return []
        
        self.metrics_collector.add_collector('system', system_metrics)
        self.metrics_collector.add_collector('audio', audio_metrics)
    
    def add_widget(self, widget: DashboardWidget):
        """Add widget to dashboard"""
        self.widgets.append(widget)
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate dashboard data for frontend
        
        Returns:
            Dashboard configuration and data
        """
        # Collect latest metrics
        metrics = await self.metrics_collector.collect_all_metrics()
        self.metrics_collector.store_metrics(metrics)
        
        # Generate widget data
        widget_data = {}
        for widget in self.widgets:
            widget_data[widget.widget_id] = await self._generate_widget_data(widget)
        
        return {
            'dashboard_id': self.dashboard_id,
            'title': self.title,
            'timestamp': datetime.utcnow().isoformat(),
            'widgets': widget_data,
            'status': 'running' if self.is_running else 'stopped'
        }
    
    async def _generate_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for specific widget"""
        widget_data = {
            'widget_id': widget.widget_id,
            'type': widget.widget_type,
            'title': widget.title,
            'data': [],
            'config': widget.config or {}
        }
        
        # Execute metric queries
        for query in widget.metric_queries:
            metric_data = self._execute_metric_query(query)
            widget_data['data'].extend(metric_data)
        
        return widget_data
    
    def _execute_metric_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute metric query
        
        Args:
            query: Metric query string (simplified format)
            
        Returns:
            Query results
        """
        # Simple query parser (in production, use PromQL or similar)
        parts = query.split()
        if len(parts) < 1:
            return []
        
        metric_name = parts[0]
        
        # Extract time range if specified
        time_range = 3600  # default 1 hour
        if 'last_hour' in query:
            time_range = 3600
        elif 'last_day' in query:
            time_range = 86400
        
        metrics = self.metrics_collector.query_metrics(metric_name, time_range)
        
        return [
            {
                'timestamp': metric.timestamp,
                'value': metric.value,
                'tags': metric.tags
            }
            for metric in metrics
        ]
    
    async def start(self):
        """Start dashboard monitoring"""
        self.is_running = True
        
        # Start periodic metric collection
        while self.is_running:
            try:
                metrics = await self.metrics_collector.collect_all_metrics()
                self.metrics_collector.store_metrics(metrics)
                await asyncio.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)  # Back off on errors
    
    def stop(self):
        """Stop dashboard monitoring"""
        self.is_running = False


class DashboardManager:
    """
    Manages multiple monitoring dashboards
    """
    
    def __init__(self):
        """Initialize dashboard manager"""
        self.dashboards = {}
        self.default_dashboard = None
        self._setup_default_dashboards()
    
    def _setup_default_dashboards(self):
        """Setup default dashboards"""
        # System overview dashboard
        system_dashboard = Dashboard('system_overview', 'AG-06 System Overview')
        
        # Add system widgets
        system_dashboard.add_widget(DashboardWidget(
            widget_id='cpu_gauge',
            widget_type='gauge',
            title='CPU Usage',
            metric_queries=['cpu_percent'],
            config={'max_value': 100, 'unit': '%', 'threshold': 80}
        ))
        
        system_dashboard.add_widget(DashboardWidget(
            widget_id='memory_gauge',
            widget_type='gauge',
            title='Memory Usage',
            metric_queries=['memory_percent'],
            config={'max_value': 100, 'unit': '%', 'threshold': 85}
        ))
        
        system_dashboard.add_widget(DashboardWidget(
            widget_id='optimizations_counter',
            widget_type='counter',
            title='Total Optimizations',
            metric_queries=['total_optimizations'],
            config={'increment': True}
        ))
        
        system_dashboard.add_widget(DashboardWidget(
            widget_id='agent_status',
            widget_type='gauge',
            title='Agent Status',
            metric_queries=['agent_status'],
            config={'max_value': 1, 'labels': ['Stopped', 'Running']}
        ))
        
        self.add_dashboard(system_dashboard)
        self.default_dashboard = system_dashboard
    
    def add_dashboard(self, dashboard: Dashboard):
        """Add dashboard to manager"""
        self.dashboards[dashboard.dashboard_id] = dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)
    
    async def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data for frontend"""
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard:
            return await dashboard.generate_dashboard_data()
        return None
    
    async def get_all_dashboards_summary(self) -> Dict[str, Any]:
        """Get summary of all dashboards"""
        summaries = {}
        
        for dashboard_id, dashboard in self.dashboards.items():
            summaries[dashboard_id] = {
                'title': dashboard.title,
                'widget_count': len(dashboard.widgets),
                'status': 'running' if dashboard.is_running else 'stopped'
            }
        
        return {
            'dashboards': summaries,
            'default_dashboard': self.default_dashboard.dashboard_id if self.default_dashboard else None,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def start_all_dashboards(self):
        """Start monitoring for all dashboards"""
        tasks = []
        for dashboard in self.dashboards.values():
            if not dashboard.is_running:
                tasks.append(asyncio.create_task(dashboard.start()))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop_all_dashboards(self):
        """Stop all dashboard monitoring"""
        for dashboard in self.dashboards.values():
            dashboard.stop()


# Create global dashboard manager
dashboard_manager = DashboardManager()

# Export dashboard components
__all__ = [
    'MetricData',
    'DashboardWidget', 
    'MetricsCollector',
    'Dashboard',
    'DashboardManager',
    'dashboard_manager'
]