"""Find what is warping the music bus.

Push a KNOWN 440 Hz sine through the exact BlackHole -> Ring -> consumer path the
app uses, then measure the recovered signal. A clean path returns 440 Hz with low
THD. Pitch error => sample-rate problem. Glitches/jumps => clock drift. Harmonics
=> clipping.
"""
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd

sys.path.insert(0, "/Users/nguythe/aioke")
from aioke_live import Ring, find_ag06, find_device

SR = 44100
F0 = 440.0
BS = 128

# 1. write a pure 440 Hz wav
n = int(SR * 6)
t = np.arange(n) / SR
sine = (0.5 * np.sin(2 * np.pi * F0 * t) * 32767).astype(np.int16)
with wave.open("/tmp/sine440.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(np.column_stack([sine, sine]).tobytes())

bh, bhinfo = find_device("blackhole")
ag, aginfo = find_ag06()
print(f"BlackHole dev={bh} sr={int(bhinfo['default_samplerate'])}")
print(f"AG06      dev={ag} sr={int(aginfo['default_samplerate'])}")

ring = Ring(1 << 16)
target = BS * 20
drift_jumps = [0]
starved = [0]
out_chunks = []


def sys_cb(indata, frames, tinfo, status):
    ring.write(np.mean(indata, axis=1).astype(np.float64))


# consumer runs on the AG06 clock, exactly like the real app
def main_cb(indata, outdata, frames, tinfo, status):
    before = ring.w - ring.r
    ring.drift_correct(target)
    if (ring.w - ring.r) != before:
        drift_jumps[0] += 1
    buf = np.zeros(frames)
    if not ring.read(buf):
        starved[0] += 1
    out_chunks.append(buf.copy())
    outdata[:] = 0


sysst = sd.InputStream(device=bh, samplerate=SR, blocksize=BS, channels=2,
                       dtype="float32", callback=sys_cb)
mainst = sd.Stream(device=(ag, ag), samplerate=SR, blocksize=BS, channels=2,
                   dtype="float32", callback=main_cb)

sysst.start()
mainst.start()
threading.Thread(
    target=lambda: subprocess.run(["afplay", "/tmp/sine440.wav"], capture_output=True),
    daemon=True,
).start()
time.sleep(5.0)
mainst.stop(); mainst.close()
sysst.stop(); sysst.close()

y = np.concatenate(out_chunks)
y = y[int(SR * 1.0):]          # drop startup
y = y[np.abs(y) > 0][: SR * 3] if np.any(np.abs(y) > 0) else y
if len(y) < SR:
    print("NO SIGNAL recovered — music never reached the ring.")
    raise SystemExit

# spectrum
w = np.hanning(len(y))
Y = np.abs(np.fft.rfft(y * w))
freqs = np.fft.rfftfreq(len(y), 1 / SR)
k = int(np.argmax(Y))
peak_hz = freqs[k]
cents = 1200 * np.log2(peak_hz / F0) if peak_hz > 0 else 0

# THD: energy in harmonics 2..5 vs fundamental
def band(f):
    i = np.argmin(np.abs(freqs - f))
    return float(np.sum(Y[max(0, i - 3): i + 4] ** 2))

fund = band(peak_hz)
harm = sum(band(peak_hz * h) for h in (2, 3, 4, 5))
thd = 100 * np.sqrt(harm / fund) if fund > 0 else 0

print()
print(f"recovered fundamental : {peak_hz:8.2f} Hz   (source 440.00 Hz)")
print(f"pitch error           : {cents:+8.1f} cents  ({100*(peak_hz/F0-1):+.2f}%)")
print(f"THD (harmonics 2-5)   : {thd:8.2f} %")
print(f"drift-correct jumps   : {drift_jumps[0]}")
print(f"ring starvations      : {starved[0]}")
print()
if abs(cents) > 20:
    print(">>> PITCH ERROR — sample-rate mismatch in the capture path.")
elif thd > 5:
    print(">>> DISTORTION — clipping or nonlinearity in the chain.")
elif drift_jumps[0] > 2:
    if thd < 1.0 and abs(cents) < 10:
        print(">>> CLOCK DRIFT — successfully mitigated by gradual zero-crossing correction.")
    else:
        print(">>> CLOCK DRIFT — ring pointer is jumping; that is the warble/stutter.")
else:
    print(">>> Music bus is CLEAN. The warping is downstream (vocal chain / output).")
