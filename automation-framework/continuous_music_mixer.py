#!/usr/bin/env python3
"""
CONTINUOUS MUSIC MIXER for AG06
Real-time audio monitoring and mixing for music playback
"""
import requests
import time
import sys
import signal
from datetime import datetime

class ContinuousMusicMixer:
    def __init__(self):
        self.api_url = 'http://localhost:5001'
        self.running = True
        self.session_start = datetime.now()
        self.peak_level = -60.0
        self.clip_count = 0
        
        # Register signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print('\n\n🛑 Stopping music mixer...')
        self.running = False
        self.show_session_stats()
        sys.exit(0)
    
    def show_session_stats(self):
        """Display session statistics"""
        duration = datetime.now() - self.session_start
        print('\n' + '='*60)
        print('📊 SESSION STATISTICS')
        print('='*60)
        print(f'  • Duration: {duration}')
        print(f'  • Peak Level: {self.peak_level:.1f} dB')
        print(f'  • Clips Detected: {self.clip_count}')
        print('='*60)
    
    def start_mixer(self):
        """Initialize music mixing session"""
        print('🎵 CONTINUOUS MUSIC MIXER')
        print('='*60)
        print('Initializing audio system...')
        
        try:
            # Start monitoring
            r = requests.post(f'{self.api_url}/api/start')
            print('✅ Audio monitoring active')
            
            # Apply music preset
            r = requests.post(f'{self.api_url}/api/preset/music')
            print('✅ Music preset applied')
            
            # Set optimal music levels
            requests.post(f'{self.api_url}/api/set_volume', json={'value': 75})
            requests.post(f'{self.api_url}/api/set_gain', json={'value': 50})
            requests.post(f'{self.api_url}/api/set_compression', json={'value': 2.0})
            requests.post(f'{self.api_url}/api/set_noise_gate', json={'value': -45})
            requests.post(f'{self.api_url}/api/set_eq', json={'low': 2, 'mid': 0, 'high': 1})
            
            print('✅ Optimal music settings configured')
            print('='*60)
            print('🎧 MUSIC MIXING ACTIVE - Press Ctrl+C to stop')
            print('='*60)
            print()
            
            return True
        except Exception as e:
            print(f'❌ Error starting mixer: {e}')
            return False
    
    def create_spectrum_bar(self, value, height=5):
        """Create a vertical spectrum bar"""
        level = int(value * height)
        if level >= 4: return '█'
        elif level >= 3: return '▓'
        elif level >= 2: return '▒'
        elif level >= 1: return '░'
        else: return ' '
    
    def monitor_continuously(self):
        """Continuous monitoring loop"""
        frame = 0
        last_status_time = time.time()
        
        while self.running:
            try:
                # Get current status
                r = requests.get(f'{self.api_url}/api/status')
                data = r.json()
                
                rms = data['input_level']['rms']
                peak = data['input_level']['peak']
                clipping = data['input_level']['clipping']
                spectrum = data.get('spectrum', [])[:8]
                
                # Update statistics
                if peak > self.peak_level:
                    self.peak_level = peak
                if clipping:
                    self.clip_count += 1
                
                # Create level meter (40 chars wide)
                meter_width = 40
                rms_normalized = max(0, min(1, (rms + 60) / 60))
                filled = int(rms_normalized * meter_width)
                
                # Color coding for levels
                if clipping:
                    meter = '🔴' * filled + '░' * (meter_width - filled)
                    status = '⚠️ CLIPPING!'
                elif rms > -10:
                    meter = '🟡' * filled + '░' * (meter_width - filled)
                    status = '♪ HOT'
                elif rms > -20:
                    meter = '🟢' * filled + '░' * (meter_width - filled)
                    status = '♪ GOOD'
                else:
                    meter = '█' * filled + '░' * (meter_width - filled)
                    status = '♪ ♫'
                
                # Create spectrum visualization
                spectrum_viz = ''
                for i, val in enumerate(spectrum):
                    bar = self.create_spectrum_bar(val)
                    spectrum_viz += bar
                
                # Format output line
                output = f'\r{meter} {rms:6.1f}dB [{spectrum_viz}] {status}  '
                
                # Add timestamp every 10 seconds
                if time.time() - last_status_time > 10:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f'\n[{timestamp}] Peak: {self.peak_level:.1f}dB, Clips: {self.clip_count}')
                    last_status_time = time.time()
                
                # Update display
                sys.stdout.write(output)
                sys.stdout.flush()
                
                frame += 1
                time.sleep(0.05)  # 20 FPS update rate
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                if frame % 100 == 0:  # Show errors occasionally
                    print(f'\n⚠️ Connection issue: {e}')
                time.sleep(0.5)
    
    def run(self):
        """Main execution"""
        if self.start_mixer():
            self.monitor_continuously()

if __name__ == '__main__':
    print('\n' + '='*60)
    print('🎛️ AG06 CONTINUOUS MUSIC MIXER')
    print('='*60)
    print('• Real-time level monitoring')
    print('• Automatic clipping detection')
    print('• 8-band spectrum analyzer')
    print('• Optimized for music playback')
    print('='*60)
    print()
    
    mixer = ContinuousMusicMixer()
    mixer.run()