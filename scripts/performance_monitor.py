#!/usr/bin/env python3
"""
Performance Monitoring Script for AG06 Mixer
Monitors latency, memory, CPU and generates alerts
"""

import json
import time
import sys
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import statistics
from dataclasses import dataclass
from enum import Enum

# Performance thresholds
LATENCY_THRESHOLD_MS = 5
LATENCY_WARNING_MS = 3
MEMORY_THRESHOLD_MB = 15
MEMORY_WARNING_MB = 12
CPU_THRESHOLD_PERCENT = 70
CPU_WARNING_PERCENT = 60
ERROR_RATE_THRESHOLD = 0.01
ERROR_RATE_WARNING = 0.005

# Monitoring configuration
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
MONITORING_INTERVAL = 30  # seconds
ALERT_COOLDOWN = 300  # seconds between same alerts

# Notification configuration
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
PAGERDUTY_ROUTING_KEY = os.getenv("PAGERDUTY_ROUTING_KEY")

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_SENDER_USER = os.getenv("EMAIL_SENDER_USER")
EMAIL_SENDER_PASSWORD = os.getenv("EMAIL_SENDER_PASSWORD")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS")  # Comma separated

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
SMS_RECIPIENTS = os.getenv("SMS_RECIPIENTS")  # Comma separated


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Performance metric data"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: str


@dataclass
class Alert:
    """Alert data structure"""
    severity: AlertSeverity
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: datetime


class PerformanceMonitor:
    """Performance monitoring system for AG06 Mixer"""
    
    def __init__(self, prometheus_url: str = PROMETHEUS_URL):
        self.prometheus_url = prometheus_url
        self.metrics_history: Dict[str, List[Metric]] = {}
        self.alerts_history: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
    def query_prometheus(self, query: str) -> Optional[Dict]:
        """Query Prometheus for metrics"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return None
    
    def get_audio_latency(self) -> List[Metric]:
        """Get audio latency metrics"""
        query = "audio_latency_ms"
        result = self.query_prometheus(query)
        
        if not result or result["status"] != "success":
            return []
        
        metrics = []
        for item in result["data"]["result"]:
            metric = Metric(
                name="audio_latency",
                value=float(item["value"][1]),
                timestamp=datetime.fromtimestamp(item["value"][0]),
                labels=item["metric"],
                unit="ms"
            )
            metrics.append(metric)
        
        return metrics
    
    def get_memory_usage(self) -> List[Metric]:
        """Get memory usage per channel"""
        query = "memory_usage_mb / channel_count"
        result = self.query_prometheus(query)
        
        if not result or result["status"] != "success":
            return []
        
        metrics = []
        for item in result["data"]["result"]:
            metric = Metric(
                name="memory_per_channel",
                value=float(item["value"][1]),
                timestamp=datetime.fromtimestamp(item["value"][0]),
                labels=item["metric"],
                unit="MB/channel"
            )
            metrics.append(metric)
        
        return metrics
    
    def get_cpu_usage(self) -> List[Metric]:
        """Get CPU usage percentage"""
        query = "cpu_usage_percent"
        result = self.query_prometheus(query)
        
        if not result or result["status"] != "success":
            return []
        
        metrics = []
        for item in result["data"]["result"]:
            metric = Metric(
                name="cpu_usage",
                value=float(item["value"][1]),
                timestamp=datetime.fromtimestamp(item["value"][0]),
                labels=item["metric"],
                unit="%"
            )
            metrics.append(metric)
        
        return metrics
    
    def get_error_rate(self) -> List[Metric]:
        """Get error rate"""
        query = "rate(http_requests_errors_total[5m])"
        result = self.query_prometheus(query)
        
        if not result or result["status"] != "success":
            return []
        
        metrics = []
        for item in result["data"]["result"]:
            metric = Metric(
                name="error_rate",
                value=float(item["value"][1]),
                timestamp=datetime.fromtimestamp(item["value"][0]),
                labels=item["metric"],
                unit="errors/sec"
            )
            metrics.append(metric)
        
        return metrics
    
    def check_thresholds(self, metric: Metric) -> Optional[Alert]:
        """Check if metric exceeds thresholds"""
        alert = None
        
        if metric.name == "audio_latency":
            if metric.value > LATENCY_THRESHOLD_MS:
                alert = Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric=metric.name,
                    message=f"Audio latency {metric.value:.2f}ms exceeds threshold",
                    value=metric.value,
                    threshold=LATENCY_THRESHOLD_MS,
                    timestamp=metric.timestamp
                )
            elif metric.value > LATENCY_WARNING_MS:
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    metric=metric.name,
                    message=f"Audio latency {metric.value:.2f}ms approaching threshold",
                    value=metric.value,
                    threshold=LATENCY_WARNING_MS,
                    timestamp=metric.timestamp
                )
        
        elif metric.name == "memory_per_channel":
            if metric.value > MEMORY_THRESHOLD_MB:
                alert = Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric=metric.name,
                    message=f"Memory usage {metric.value:.2f}MB/channel exceeds threshold",
                    value=metric.value,
                    threshold=MEMORY_THRESHOLD_MB,
                    timestamp=metric.timestamp
                )
            elif metric.value > MEMORY_WARNING_MB:
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    metric=metric.name,
                    message=f"Memory usage {metric.value:.2f}MB/channel approaching threshold",
                    value=metric.value,
                    threshold=MEMORY_WARNING_MB,
                    timestamp=metric.timestamp
                )
        
        elif metric.name == "cpu_usage":
            if metric.value > CPU_THRESHOLD_PERCENT:
                alert = Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric=metric.name,
                    message=f"CPU usage {metric.value:.1f}% exceeds threshold",
                    value=metric.value,
                    threshold=CPU_THRESHOLD_PERCENT,
                    timestamp=metric.timestamp
                )
            elif metric.value > CPU_WARNING_PERCENT:
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    metric=metric.name,
                    message=f"CPU usage {metric.value:.1f}% approaching threshold",
                    value=metric.value,
                    threshold=CPU_WARNING_PERCENT,
                    timestamp=metric.timestamp
                )
        
        elif metric.name == "error_rate":
            if metric.value > ERROR_RATE_THRESHOLD:
                alert = Alert(
                    severity=AlertSeverity.CRITICAL,
                    metric=metric.name,
                    message=f"Error rate {metric.value:.4f} exceeds threshold",
                    value=metric.value,
                    threshold=ERROR_RATE_THRESHOLD,
                    timestamp=metric.timestamp
                )
            elif metric.value > ERROR_RATE_WARNING:
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    metric=metric.name,
                    message=f"Error rate {metric.value:.4f} approaching threshold",
                    value=metric.value,
                    threshold=ERROR_RATE_WARNING,
                    timestamp=metric.timestamp
                )
        
        return alert
    
    def should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on cooldown"""
        alert_key = f"{alert.metric}:{alert.severity.value}"
        
        if alert_key not in self.last_alert_time:
            return True
        
        time_since_last = (alert.timestamp - self.last_alert_time[alert_key]).total_seconds()
        return time_since_last >= ALERT_COOLDOWN
    
    def send_alert(self, alert: Alert):
        """Send alert (implement notification logic here)"""
        # Color codes for terminal output
        colors = {
            AlertSeverity.INFO: "\033[36m",      # Cyan
            AlertSeverity.WARNING: "\033[33m",   # Yellow
            AlertSeverity.CRITICAL: "\033[31m"   # Red
        }
        reset = "\033[0m"
        
        color = colors.get(alert.severity, "")
        icon = "ℹ️" if alert.severity == AlertSeverity.INFO else "⚠️" if alert.severity == AlertSeverity.WARNING else "🚨"
        
        print(f"{color}{icon} [{alert.severity.value.upper()}] {alert.message}{reset}")
        print(f"   Value: {alert.value:.2f}, Threshold: {alert.threshold:.2f}")
        print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update last alert time
        alert_key = f"{alert.metric}:{alert.severity.value}"
        self.last_alert_time[alert_key] = alert.timestamp
        
        # Add to history
        self.alerts_history.append(alert)
        
        # Send notifications
        if SLACK_WEBHOOK_URL:
            self.send_slack_alert(alert)

        if PAGERDUTY_ROUTING_KEY:
            self.send_pagerduty_alert(alert)

        if EMAIL_SENDER_USER and EMAIL_RECIPIENTS:
            self.send_email_alert(alert)

        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and SMS_RECIPIENTS:
            self.send_sms_alert(alert)
    
    def send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        if not SLACK_WEBHOOK_URL:
            return

        try:
            color = "#36a64f" if alert.severity == AlertSeverity.INFO else "#ffcc00" if alert.severity == AlertSeverity.WARNING else "#ff0000"
            payload = {
                "text": f"*{alert.severity.value.upper()}*: {alert.message}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {"title": "Metric", "value": alert.metric, "short": True},
                            {"title": "Value", "value": str(alert.value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                        ]
                    }
                ]
            }
            requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")

    def send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty"""
        if not PAGERDUTY_ROUTING_KEY:
            return

        try:
            payload = {
                "routing_key": PAGERDUTY_ROUTING_KEY,
                "event_action": "trigger",
                "dedup_key": f"{alert.metric}_{alert.timestamp.strftime('%Y%m%d%H%M')}",
                "payload": {
                    "summary": f"{alert.severity.value.upper()}: {alert.message}",
                    "severity": "critical" if alert.severity == AlertSeverity.CRITICAL else "warning" if alert.severity == AlertSeverity.WARNING else "info",
                    "source": "ag06-mixer",
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": {
                        "metric": alert.metric,
                        "value": alert.value,
                        "threshold": alert.threshold
                    }
                }
            }
            requests.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=10)
        except Exception as e:
            print(f"Failed to send PagerDuty alert: {e}")

    def send_email_alert(self, alert: Alert):
        """Send alert via Email"""
        if not (EMAIL_SENDER_USER and EMAIL_RECIPIENTS):
            return

        try:
            recipients = [r.strip() for r in EMAIL_RECIPIENTS.split(",")]
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER_USER
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"AG06 Alert: {alert.severity.value.upper()} - {alert.metric}"

            body = f"""
AG06 Mixer Performance Alert

Severity: {alert.severity.value.upper()}
Metric: {alert.metric}
Message: {alert.message}
Value: {alert.value}
Threshold: {alert.threshold}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            """
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
            server.starttls()
            if EMAIL_SENDER_PASSWORD:
                server.login(EMAIL_SENDER_USER, EMAIL_SENDER_PASSWORD)
            server.sendmail(EMAIL_SENDER_USER, recipients, msg.as_string())
            server.quit()
        except Exception as e:
            print(f"Failed to send Email alert: {e}")

    def send_sms_alert(self, alert: Alert):
        """Send alert via SMS (Twilio)"""
        if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and SMS_RECIPIENTS):
            return

        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            recipients = [r.strip() for r in SMS_RECIPIENTS.split(",")]

            message = f"AG06 Alert [{alert.severity.value.upper()}]: {alert.message}. Value: {alert.value:.2f}"

            for recipient in recipients:
                data = {
                    "From": TWILIO_FROM_NUMBER,
                    "To": recipient,
                    "Body": message
                }
                requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
        except Exception as e:
            print(f"Failed to send SMS alert: {e}")

    def calculate_statistics(self, metric_name: str, window_minutes: int = 60) -> Dict:
        """Calculate statistics for a metric over time window"""
        if metric_name not in self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history[metric_name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) > 100 else max(values)
        }
    
    def print_dashboard(self):
        """Print performance dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("AG06 MIXER PERFORMANCE DASHBOARD".center(80))
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # Current metrics
        print("\n📊 CURRENT METRICS:")
        
        for metric_name in ["audio_latency", "memory_per_channel", "cpu_usage", "error_rate"]:
            if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                latest = self.metrics_history[metric_name][-1]
                
                # Determine status indicator
                if metric_name == "audio_latency":
                    status = "✅" if latest.value <= LATENCY_WARNING_MS else "⚠️" if latest.value <= LATENCY_THRESHOLD_MS else "❌"
                elif metric_name == "memory_per_channel":
                    status = "✅" if latest.value <= MEMORY_WARNING_MB else "⚠️" if latest.value <= MEMORY_THRESHOLD_MB else "❌"
                elif metric_name == "cpu_usage":
                    status = "✅" if latest.value <= CPU_WARNING_PERCENT else "⚠️" if latest.value <= CPU_THRESHOLD_PERCENT else "❌"
                else:  # error_rate
                    status = "✅" if latest.value <= ERROR_RATE_WARNING else "⚠️" if latest.value <= ERROR_RATE_THRESHOLD else "❌"
                
                print(f"  {status} {metric_name:20s}: {latest.value:8.2f} {latest.unit}")
        
        # Statistics
        print("\n📈 STATISTICS (Last 60 minutes):")
        
        for metric_name in ["audio_latency", "memory_per_channel", "cpu_usage", "error_rate"]:
            stats = self.calculate_statistics(metric_name, 60)
            if stats:
                print(f"\n  {metric_name}:")
                print(f"    Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
                print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                print(f"    P95: {stats['p95']:.2f}, P99: {stats['p99']:.2f}")
        
        # Recent alerts
        print("\n🔔 RECENT ALERTS:")
        recent_alerts = self.alerts_history[-5:] if self.alerts_history else []
        if recent_alerts:
            for alert in recent_alerts:
                icon = "⚠️" if alert.severity == AlertSeverity.WARNING else "🚨"
                print(f"  {icon} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.message}")
        else:
            print("  ✅ No recent alerts")
        
        print("\n" + "=" * 80)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("Starting AG06 Mixer Performance Monitor...")
        print(f"Prometheus URL: {self.prometheus_url}")
        print(f"Monitoring interval: {MONITORING_INTERVAL} seconds")
        print("-" * 80)
        
        while True:
            try:
                # Collect metrics
                metrics = []
                metrics.extend(self.get_audio_latency())
                metrics.extend(self.get_memory_usage())
                metrics.extend(self.get_cpu_usage())
                metrics.extend(self.get_error_rate())
                
                # Store metrics in history
                for metric in metrics:
                    if metric.name not in self.metrics_history:
                        self.metrics_history[metric.name] = []
                    self.metrics_history[metric.name].append(metric)
                    
                    # Keep only last 24 hours of data
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.metrics_history[metric.name] = [
                        m for m in self.metrics_history[metric.name]
                        if m.timestamp >= cutoff_time
                    ]
                    
                    # Check thresholds and send alerts
                    alert = self.check_thresholds(metric)
                    if alert and self.should_send_alert(alert):
                        self.send_alert(alert)
                
                # Print dashboard
                self.print_dashboard()
                
                # Sleep until next iteration
                time.sleep(MONITORING_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped by user.")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(MONITORING_INTERVAL)
    
    def export_metrics(self, output_file: str):
        """Export metrics history to JSON file"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": []
        }
        
        # Export metrics
        for metric_name, metrics in self.metrics_history.items():
            export_data["metrics"][metric_name] = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "labels": m.labels,
                    "unit": m.unit
                }
                for m in metrics
            ]
        
        # Export alerts
        export_data["alerts"] = [
            {
                "severity": a.severity.value,
                "metric": a.metric,
                "message": a.message,
                "value": a.value,
                "threshold": a.threshold,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self.alerts_history
        ]
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Metrics exported to {output_file}")


def main():
    """Main entry point"""
    global MONITORING_INTERVAL
    import argparse
    
    parser = argparse.ArgumentParser(description="AG06 Mixer Performance Monitor")
    parser.add_argument(
        "--prometheus-url",
        default=PROMETHEUS_URL,
        help="Prometheus server URL"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=MONITORING_INTERVAL,
        help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--export",
        help="Export metrics to JSON file and exit"
    )
    
    args = parser.parse_args()
    
    # Override global settings
    MONITORING_INTERVAL = args.interval
    
    monitor = PerformanceMonitor(prometheus_url=args.prometheus_url)
    
    if args.export:
        # Export mode - collect metrics once and export
        metrics = []
        metrics.extend(monitor.get_audio_latency())
        metrics.extend(monitor.get_memory_usage())
        metrics.extend(monitor.get_cpu_usage())
        metrics.extend(monitor.get_error_rate())
        
        for metric in metrics:
            if metric.name not in monitor.metrics_history:
                monitor.metrics_history[metric.name] = []
            monitor.metrics_history[metric.name].append(metric)
        
        monitor.export_metrics(args.export)
    else:
        # Monitoring mode
        monitor.monitor_loop()


if __name__ == "__main__":
    main()