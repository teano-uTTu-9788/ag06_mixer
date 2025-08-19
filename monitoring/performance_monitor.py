#!/usr/bin/env python3
import time
import psutil
import json
from datetime import datetime

def monitor_performance():
    """Continuous performance monitoring"""
    while True:
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        with open('performance_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 {metrics['timestamp']}: CPU {metrics['cpu_percent']:.1f}%, Memory {metrics['memory_percent']:.1f}%")
        time.sleep(5)

if __name__ == "__main__":
    monitor_performance()
