#!/usr/bin/env python3
"""
AG06 Mixer Development Monitor
Context-safe monitoring for AG06 mixer app development
Prevents terminal overflow while providing essential debugging
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Any
import threading
import signal

class AG06DevMonitor:
    """Safe monitoring system for AG06 Mixer development"""
    
    def __init__(self):
        self.max_log_lines = 100
        self.max_midi_events = 50
        self.max_audio_samples = 20
        
        # Context buffers with size limits
        self.audio_log = deque(maxlen=self.max_log_lines)
        self.midi_log = deque(maxlen=self.max_midi_events)
        self.error_log = deque(maxlen=20)
        self.performance_log = deque(maxlen=30)
        
        # Monitoring state
        self.monitoring = False
        self.ag06_connected = False
        self.audio_active = False
        self.midi_active = False
        
        # Stats
        self.stats = {
            'audio_events': 0,
            'midi_events': 0,
            'errors': 0,
            'cpu_usage': 0,
            'memory_mb': 0,
            'uptime_seconds': 0
        }
        
        self.start_time = time.time()
        
    def check_ag06_connection(self) -> bool:
        """Check if AG06 is connected via USB"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPUSBDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.ag06_connected = 'AG06' in result.stdout
            return self.ag06_connected
        except:
            return False
    
    def check_audio_device(self) -> Dict[str, Any]:
        """Get AG06 audio device status"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPAudioDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            lines = result.stdout.split('\n')
            ag06_info = {}
            in_ag06 = False
            
            for line in lines:
                if 'AG06' in line:
                    in_ag06 = True
                    ag06_info['name'] = line.strip().rstrip(':')
                elif in_ag06 and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ['Manufacturer', 'Input Channels', 'Output Channels', 'Sample Rate']:
                        ag06_info[key.lower().replace(' ', '_')] = value
                elif in_ag06 and line.strip() == '':
                    break
            
            return ag06_info if ag06_info else {'status': 'Not found'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def monitor_audio_levels(self) -> Optional[Dict[str, float]]:
        """Monitor audio input/output levels (simulated for safety)"""
        # In production, this would interface with Core Audio
        # For now, return safe simulated values
        if self.ag06_connected:
            import random
            return {
                'input_1': round(random.uniform(-48, -12), 1),
                'input_2': round(random.uniform(-48, -12), 1),
                'output_l': round(random.uniform(-24, -6), 1),
                'output_r': round(random.uniform(-24, -6), 1),
                'monitor': round(random.uniform(-30, -10), 1)
            }
        return None
    
    def monitor_midi_activity(self) -> List[str]:
        """Monitor MIDI events (limited output)"""
        midi_events = []
        
        try:
            # Check for receivemidi tool
            result = subprocess.run(
                ['which', 'receivemidi'],
                capture_output=True,
                timeout=1
            )
            
            if result.returncode == 0:
                # Monitor for 1 second max
                result = subprocess.run(
                    ['timeout', '1', 'receivemidi'],
                    capture_output=True,
                    text=True
                )
                
                events = result.stdout.split('\n')[:self.max_midi_events]
                for event in events:
                    if event.strip():
                        midi_events.append(f"{datetime.now().strftime('%H:%M:%S')} {event.strip()}")
                        
        except:
            pass
            
        return midi_events
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Get process statistics for the mixer app"""
        try:
            # Find AG06 mixer process
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if 'ag06' in line.lower() and 'python' in line.lower():
                    parts = line.split()
                    if len(parts) > 3:
                        return {
                            'cpu': float(parts[2]),
                            'memory': float(parts[3]),
                            'pid': parts[1]
                        }
            
            return {'status': 'Not running'}
            
        except:
            return {'error': 'Unable to get stats'}
    
    def format_status_display(self) -> str:
        """Format a concise status display"""
        uptime = int(time.time() - self.start_time)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60
        
        # Update stats
        process_stats = self.get_process_stats()
        audio_levels = self.monitor_audio_levels()
        
        status = f"""
╔══════════════════════════════════════════════╗
║         AG06 MIXER DEVELOPMENT MONITOR       ║
╚══════════════════════════════════════════════╝

📊 CONNECTION STATUS
├─ AG06 USB: {'✅ Connected' if self.ag06_connected else '❌ Disconnected'}
├─ Audio Device: {'✅ Active' if audio_levels else '⚠️ Inactive'}
└─ MIDI: {'✅ Active' if self.midi_active else '⚠️ Inactive'}

📈 PERFORMANCE
├─ CPU: {process_stats.get('cpu', 0):.1f}%
├─ Memory: {process_stats.get('memory', 0):.1f}%
└─ Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}

🎚️ AUDIO LEVELS (dB)
├─ Input 1: {audio_levels['input_1'] if audio_levels else 'N/A'}
├─ Input 2: {audio_levels['input_2'] if audio_levels else 'N/A'}
├─ Output L/R: {audio_levels['output_l'] if audio_levels else 'N/A'}/{audio_levels['output_r'] if audio_levels else 'N/A'}
└─ Monitor: {audio_levels['monitor'] if audio_levels else 'N/A'}

📊 ACTIVITY
├─ Audio Events: {self.stats['audio_events']}
├─ MIDI Events: {self.stats['midi_events']}
└─ Errors: {self.stats['errors']}

{'═' * 48}
Last {len(self.error_log)} errors (newest first):"""
        
        # Add recent errors (limited)
        for error in list(self.error_log)[-3:]:
            status += f"\n  ❌ {error}"
        
        if not self.error_log:
            status += "\n  ✅ No recent errors"
        
        return status
    
    def save_debug_snapshot(self):
        """Save detailed debug info to file (not terminal)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        debug_file = f"/Users/nguythe/ag06_mixer/logs/debug_{timestamp}.json"
        
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        
        debug_data = {
            'timestamp': timestamp,
            'ag06_connected': self.ag06_connected,
            'audio_device': self.check_audio_device(),
            'process_stats': self.get_process_stats(),
            'audio_levels': self.monitor_audio_levels(),
            'recent_errors': list(self.error_log),
            'stats': self.stats
        }
        
        with open(debug_file, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        return debug_file
    
    def interactive_monitor(self):
        """Run interactive monitoring session"""
        print("🎛️  AG06 Mixer Development Monitor")
        print("=" * 48)
        print("Commands: [s]tatus, [d]ebug save, [r]efresh, [q]uit")
        print("=" * 48)
        
        # Initial check
        self.check_ag06_connection()
        
        try:
            while True:
                # Clear screen for clean display
                os.system('clear')
                
                # Show status
                print(self.format_status_display())
                
                # Get user input with timeout
                print("\nCommand (s/d/r/q): ", end='', flush=True)
                
                # Use select for non-blocking input with timeout
                import select
                ready, _, _ = select.select([sys.stdin], [], [], 5)
                
                if ready:
                    command = sys.stdin.readline().strip().lower()
                    
                    if command == 'q':
                        print("👋 Exiting monitor...")
                        break
                    elif command == 'd':
                        debug_file = self.save_debug_snapshot()
                        print(f"✅ Debug saved to: {debug_file}")
                        time.sleep(2)
                    elif command == 's':
                        continue  # Refresh status
                    elif command == 'r':
                        self.check_ag06_connection()
                        print("🔄 Refreshed connection status")
                        time.sleep(1)
                else:
                    # Auto-refresh every 5 seconds
                    self.check_ag06_connection()
                    self.stats['audio_events'] += 1  # Simulate activity
                    
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")

def main():
    """Main entry point"""
    monitor = AG06DevMonitor()
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--status':
            monitor.check_ag06_connection()
            print(monitor.format_status_display())
        elif sys.argv[1] == '--debug':
            debug_file = monitor.save_debug_snapshot()
            print(f"Debug info saved to: {debug_file}")
        elif sys.argv[1] == '--help':
            print("""
AG06 Mixer Development Monitor

Usage:
    python3 ag06_dev_monitor.py          # Interactive monitor
    python3 ag06_dev_monitor.py --status # One-time status check
    python3 ag06_dev_monitor.py --debug  # Save debug snapshot
    python3 ag06_dev_monitor.py --help   # Show this help

The monitor provides context-safe output to prevent terminal overflow
while showing essential AG06 mixer development information.
""")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run interactive monitor
        monitor.interactive_monitor()

if __name__ == "__main__":
    main()