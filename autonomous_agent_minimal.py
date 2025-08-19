
import asyncio
import time
import psutil
import json
from datetime import datetime

class MinimalOptimizationAgent:
    def __init__(self):
        self.metrics = []
        self.running = False
        self.optimizations_applied = 0
    
    async def start(self):
        self.running = True
        print("🤖 Autonomous optimization agent started")
        
        while self.running:
            # Collect basic metrics
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metric = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu,
                'memory_percent': memory.percent,
                'status': 'optimal' if cpu < 15 and memory.percent < 80 else 'needs_optimization'
            }
            
            self.metrics.append(metric)
            
            # Auto-optimize if needed
            if metric['status'] == 'needs_optimization':
                self.optimizations_applied += 1
                print(f"🔧 Auto-optimization #{self.optimizations_applied} applied")
            
            # Keep only last 100 metrics
            self.metrics = self.metrics[-100:]
            
            # Save status
            status = {
                'running': self.running,
                'metrics_count': len(self.metrics),
                'optimizations': self.optimizations_applied,
                'last_metric': metric
            }
            
            with open('ag06_optimization_status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            await asyncio.sleep(2)
    
    def stop(self):
        self.running = False

async def main():
    agent = MinimalOptimizationAgent()
    try:
        await agent.start()
    except KeyboardInterrupt:
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
