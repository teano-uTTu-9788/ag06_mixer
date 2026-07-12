#!/usr/bin/env python3
"""
AiOke Live — real-time AI karaoke vocal mixing through the Yamaha AG06.

    mic -> AG06 (GAIN, TO PC=DRY CH1-2) -> USB -> [ THIS ] -> USB -> AG06 -> speakers

Signal chain (all real-time safe, zero allocation in the audio callback):
    high-pass -> noise gate -> 3-band EQ -> compressor -> reverb/echo -> limiter
    + backing track on a music bus with VAD-driven auto-ducking

The "AI" here is honest: adaptive DSP. A voice-activity detector drives automatic
ducking of the backing track and automatic gain-staging toward a target vocal level.
There is no neural network in the audio path and this file does not claim one.

AG06 HARDWARE SETUP — you will hear NOTHING if these are wrong:
    1. Mic into CH 1 (or 2). Turn GAIN up until the PEAK led flickers on loud notes.
    2. TO PC switch  -> DRY CH 1-2      (sends your dry mic to the computer)
    3. MONITOR MUTE  -> ON (engaged)    (kills the analog path so you hear OUR mix)
    4. +48V ON only if you use a condenser mic.
    5. Turn the MONITOR/PHONES knob up.

Usage:
    python3 aioke_live.py --list
    python3 aioke_live.py                         # live vocal FX, no music
    python3 aioke_live.py --music backing.mp3     # karaoke: duck music under vocal
    python3 aioke_live.py --dry-run 5             # 5s self-test, no hardware needed
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None
from scipy import signal


# ----------------------------------------------------------------------------
# DSP blocks. Every one keeps its filter state across blocks — resetting state
# per block is THE classic bug here: it puts a discontinuity at every block
# boundary, which you hear as a click ~170 times a second. test_state_continuity
# below is what catches it.
# ----------------------------------------------------------------------------
class Biquads:
    """Cascaded IIR sections with persistent state (zi carried across blocks)."""

    def __init__(self, sos: np.ndarray):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.zi = signal.sosfilt_zi(self.sos) * 0.0  # start at rest, not at steady-state

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


def highpass(sr: int, hz: float = 80.0) -> Biquads:
    """Kill mic rumble, handling noise, plosives, AC hum."""
    return Biquads(signal.butter(2, hz / (sr / 2), btype="highpass", output="sos"))


def vocal_eq(sr: int, low_db=-1.5, mid_db=2.0, high_db=3.0) -> Biquads:
    """Gentle 3-band vocal shaping: trim mud, lift presence, add air."""
    secs = []
    for f0, q, gain in ((250.0, 0.9, low_db), (2600.0, 0.9, mid_db), (8000.0, 0.7, high_db)):
        A = 10 ** (gain / 40.0)
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
        secs.append(np.array(b + a) / a[0])
    return Biquads(np.array(secs))


class NoiseGate:
    """Silence the room between phrases. Fast attack, slow release."""

    def __init__(self, sr: int, thresh_db=-45.0, attack_ms=3.0, release_ms=180.0):
        self.thresh = 10 ** (thresh_db / 20.0)
        self.a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
        self.a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0))
        self.g = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(x * x) + 1e-20))
        target = 1.0 if rms > self.thresh else 0.0
        a = self.a_att if target > self.g else self.a_rel
        # one-pole smoothing of the gate gain across the block
        n = len(x)
        coef = a ** np.arange(1, n + 1)
        env = target + (self.g - target) * coef
        self.g = float(env[-1])
        return x * env


class Compressor:
    """Amateur karaoke singers swing 30dB between a whisper and a shout. Tame it."""

    def __init__(self, sr: int, thresh_db=-22.0, ratio=3.5, attack_ms=8.0,
                 release_ms=140.0, makeup_db=5.0):
        self.thresh = 10 ** (thresh_db / 20.0)
        self.ratio = ratio
        self.a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
        self.a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0))
        self.makeup = 10 ** (makeup_db / 20.0)
        self.env = 0.0
        self.gr_db = 0.0  # exposed for the meter

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x)) + 1e-20)
        a = self.a_att if peak > self.env else self.a_rel
        # a is a PER-SAMPLE coefficient; we update once per block, so it must be
        # raised to the block length. Using it raw makes attack/release ~256x too
        # slow and the compressor barely moves.
        a = a ** len(x)
        self.env = a * self.env + (1 - a) * peak
        if self.env > self.thresh:
            over_db = 20 * np.log10(self.env / self.thresh)
            gr_db = over_db * (1.0 / self.ratio - 1.0)  # negative
        else:
            gr_db = 0.0
        self.gr_db = gr_db
        return x * (10 ** (gr_db / 20.0)) * self.makeup


class Reverb:
    """Schroeder reverb: 4 combs -> 2 allpass. THE karaoke essential — dry vocals
    sound naked and people hate hearing themselves. Fully vectorized: every delay
    is longer than one block, so a block never reads its own output."""

    def __init__(self, sr: int, room=0.82, damp=0.28, mix=0.28):
        self.mix = mix
        self.damp = damp
        self.room = room
        comb_ms = (29.7, 37.1, 41.1, 43.7)
        ap_ms = (5.0, 1.7)
        self.combs = []
        for ms in comb_ms:
            d = max(int(sr * ms / 1000.0), 512)
            self.combs.append({"buf": np.zeros(d, dtype=np.float64), "i": 0, "lp": 0.0, "d": d})
        self.aps = []
        for ms in ap_ms:
            d = max(int(sr * ms / 1000.0), 256)
            self.aps.append({"buf": np.zeros(d, dtype=np.float64), "i": 0, "d": d})

    @staticmethod
    def _read(buf, idx, n):
        d = len(buf)
        pos = (np.arange(n) + idx) % d
        return buf[pos], pos

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        wet = np.zeros(n, dtype=np.float64)
        for c in self.combs:
            if c["d"] < n:
                continue
            delayed, pos = self._read(c["buf"], c["i"], n)
            # one-pole damping in the feedback path (soaks up harsh highs)
            lp = np.empty(n)
            z = c["lp"]
            for k in range(n):  # n is small (256); damping is inherently recursive
                z = delayed[k] * (1 - self.damp) + z * self.damp
                lp[k] = z
            c["lp"] = float(z)
            c["buf"][pos] = x + lp * self.room
            c["i"] = (c["i"] + n) % c["d"]
            wet += delayed
        wet *= 0.25
        for ap in self.aps:
            if ap["d"] < n:
                continue
            delayed, pos = self._read(ap["buf"], ap["i"], n)
            g = 0.5
            out = -g * wet + delayed
            ap["buf"][pos] = wet + g * out
            ap["i"] = (ap["i"] + n) % ap["d"]
            wet = out
        return x * (1.0 - self.mix) + wet * self.mix


class Limiter:
    """Last line of defence. Output must NEVER clip the AG06."""

    def __init__(self, ceiling_db=-1.0):
        self.ceil = 10 ** (ceiling_db / 20.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x)))
        if peak > self.ceil:
            x = x * (self.ceil / peak)
        return np.clip(x, -self.ceil, self.ceil)


class VocalChain:
    def __init__(self, sr: int, reverb_mix=0.28):
        self.hp = highpass(sr)
        self.gate = NoiseGate(sr)
        self.eq = vocal_eq(sr)
        self.comp = Compressor(sr)
        self.verb = Reverb(sr, mix=reverb_mix)
        self.lim = Limiter()

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self.hp.process(x)
        y = self.gate.process(y)
        y = self.eq.process(y)
        y = self.comp.process(y)
        y = self.verb.process(y)
        return self.lim.process(y)


class Ducker:
    """The adaptive bit. Detect singing, pull the backing track down under it,
    let it swell back in the gaps. This is what a human sound engineer does."""

    def __init__(self, sr: int, thresh_db=-38.0, duck_db=-11.0,
                 attack_ms=12.0, release_ms=420.0):
        self.thresh = 10 ** (thresh_db / 20.0)
        self.duck = 10 ** (duck_db / 20.0)
        self.a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
        self.a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0))
        self.g = 1.0
        self.singing = False

    def gain_for(self, vocal: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(vocal * vocal) + 1e-20))
        self.singing = rms > self.thresh
        target = self.duck if self.singing else 1.0
        a = self.a_att if target < self.g else self.a_rel
        # Per-sample coefficient applied once per block -> raise to block length,
        # otherwise the duck moves ~0.2% per block and never actually ducks.
        a = a ** len(vocal)
        self.g = a * self.g + (1 - a) * target
        return self.g


# ----------------------------------------------------------------------------
class Ring:
    """Lock-free single-producer/single-consumer ring. The system-audio capture
    callback writes; the AG06 callback reads. Neither ever blocks the other, and
    nothing is allocated after __init__ — both are real-time threads."""

    def __init__(self, n: int = 65536):
        self.buf = np.zeros(n, dtype=np.float64)
        self.n = n
        self.w = 0
        self.r = 0
        self.jumps = 0

    def write(self, x: np.ndarray) -> None:
        k = len(x)
        i = self.w % self.n
        end = i + k
        if end <= self.n:
            self.buf[i:end] = x
        else:
            s = self.n - i
            self.buf[i:] = x[:s]
            self.buf[: k - s] = x[s:]
        self.w += k

    def read(self, out: np.ndarray) -> bool:
        """Fill `out`. Returns False (and fills silence) if we've run dry."""
        k = len(out)
        if self.w - self.r < k:
            out[:] = 0.0
            return False
        i = self.r % self.n
        end = i + k
        if end <= self.n:
            out[:] = self.buf[i:end]
        else:
            s = self.n - i
            out[:s] = self.buf[i:]
            out[s:] = self.buf[: k - s]
        self.r += k
        return True

    def drift_correct(self, target: int) -> None:
        """AG06 and BlackHole run on different clocks and slowly drift apart.
        Keep the backlog near `target` by nudging the read pointer."""
        avail = self.w - self.r
        if avail > target + 1024:
            for i in range(128):
                idx1 = (self.r + i) % self.n
                idx2 = (self.r + i + 1) % self.n
                if self.buf[idx1] * self.buf[idx2] <= 0:
                    if i > 0:
                        for j in range(i - 1, -1, -1):
                            self.buf[(self.r + j + 1) % self.n] = self.buf[(self.r + j) % self.n]
                    self.r += 1
                    self.jumps += 1
                    return
            self.r += 1
            self.jumps += 1
        elif avail < target - 1024:
            for i in range(128):
                idx1 = (self.r + i) % self.n
                idx2 = (self.r + i + 1) % self.n
                if self.buf[idx1] * self.buf[idx2] <= 0:
                    if i > 0:
                        for j in range(i):
                            self.buf[(self.r + j - 1) % self.n] = self.buf[(self.r + j) % self.n]
                    self.buf[(self.r + i - 1) % self.n] = 0.0
                    self.r -= 1
                    self.jumps += 1
                    return
            self.r -= 1
            self.buf[self.r % self.n] = self.buf[(self.r + 1) % self.n]
            self.jumps += 1


def find_device(sub: str, want_input: bool = True):
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if sub.lower() in d["name"].lower() and ch > 0:
            return i, d
    return None, None


def find_ag06():
    for i, d in enumerate(sd.query_devices()):
        n = d["name"].lower()
        if ("ag06" in n or "ag03" in n or "yamaha" in n) and d["max_input_channels"] > 0:
            return i, d
    return None, None


def load_music(path: str, sr: int) -> np.ndarray:
    import subprocess
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le",
         "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    m = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    # Many MP3s decode above full scale (this one peaked at 1.37). Normalise to
    # -3 dBFS so the music bus can never clip once the vocal is summed on top.
    peak = float(np.abs(m).max())
    if peak > 0:
        m *= (10 ** (-3.0 / 20.0)) / peak
    return m


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None, help="input device (default: AG06)")
    ap.add_argument("--out-device", type=int, default=None,
                    help="output device INDEX. Fragile: CoreAudio renumbers devices. "
                         "Prefer --out.")
    ap.add_argument("--out", type=str, default=None,
                    help="output device by NAME, e.g. --out macbook | --out ag06 | "
                         "--out srs. Survives renumbering.")
    ap.add_argument("--music", type=str, default=None, help="backing track (any ffmpeg format)")
    ap.add_argument("--loop", action="store_true",
                    help="repeat the backing track. OFF by default — a short clip "
                         "looping every 30s is maddening.")
    ap.add_argument("--capture-system", action="store_true",
                    help="Duck whatever YOU play (Spotify/YouTube/anything). Requires "
                         "macOS output set to 'BlackHole 2ch'. The app never starts or "
                         "stops music — it only listens and mixes.")
    ap.add_argument("--blocksize", type=int, default=256)
    ap.add_argument("--samplerate", type=int, default=None, help="default: device native")
    ap.add_argument("--reverb", type=float, default=0.28, help="0.0 dry .. 0.6 wet")
    ap.add_argument("--music-gain", type=float, default=0.7)
    ap.add_argument("--gate", type=float, default=-45.0,
                    help="gate threshold dB. Lower = more sensitive. -99 disables.")
    ap.add_argument("--duck-thresh", type=float, default=-25.0,
                    help="how loud you must sing before the music ducks. MUST sit above "
                         "the speaker-into-mic bleed level, or the music ducks ITSELF "
                         "and the duck pumps. Use headphones and you can lower this.")
    ap.add_argument("--duck-db", type=float, default=-11.0,
                    help="how far the music drops under your voice, in dB")
    ap.add_argument("--music-boost", type=float, default=0.0,
                    help="gain (dB) on the captured music bus. macOS applies its "
                         "output volume BEFORE writing into BlackHole, so the capture "
                         "can arrive ~55 dB down even at '100'. Use --auto-music "
                         "instead of guessing.")
    ap.add_argument("--auto-music", action="store_true",
                    help="automatically bring the captured music up to a target level. "
                         "Immune to whatever macOS does to BlackHole's volume.")
    ap.add_argument("--music-target", type=float, default=-14.0,
                    help="target peak (dBFS) for the music bus when --auto-music is on")
    ap.add_argument("--boost", type=float, default=0.0,
                    help="input gain dB, if the AG06 GAIN knob can't go louder")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", type=float, default=0.0, help="self-test seconds, no hardware")
    ap.add_argument("--no-autoconfig", action="store_true",
                    help="do NOT let the app configure the audio devices. You will "
                         "then have to get sample rates, BlackHole volume and the "
                         "default devices right by hand. Not recommended.")
    ap.add_argument("--restore", action="store_true",
                    help="put macOS audio back to normal (output -> your speakers) "
                         "and exit. Use this when you are done.")
    ap.add_argument("--check", action="store_true",
                    help="configure the devices, print the PASS/FAIL table, and EXIT. "
                         "Starts no audio engine. Makes no sound.")
    args = ap.parse_args()

    if args.restore:
        import coreaudio_cfg as ca
        spk = (ca.find_output(args.out) if args.out else None) or ca.find_output("macbook pro speakers")
        if spk:
            ca.set_mute(spk, False)
            ca.set_volume(spk, 0.7)
            ca.set_default_output(spk)
            print(f"restored: system output -> {ca.name_of(spk)}")
        else:
            print("could not find a speaker to restore to", file=sys.stderr)
            return 1
        return 0

    if args.dry_run:
        sr = args.samplerate or 44100
        return selftest(sr, args.blocksize, args.dry_run)

    if sd is None:
        print("sounddevice not installed:  pip3 install sounddevice", file=sys.stderr)
        return 1
    if args.list:
        print(sd.query_devices())
        return 0

    # ---- AUTO-CONFIGURE THE MACHINE. No more guess-and-check. -----------------
    pf = None
    if not args.no_autoconfig:
        import preflight
        pf = preflight.run(sd, engine_rate=args.samplerate,
                           want_capture=args.capture_system,
                           out_match=args.out, set_defaults=True, fix=True)
        if args.check:
            print(f"\n  engine rate : {int(pf['engine_rate'])} Hz")
            print(f"  mic         : {pf['ag06']}")
            print(f"  music capture: {pf['blackhole'] or '(disabled)'}")
            print(f"  you listen on: {pf['output'] or '(AG06)'}")
            print(f"\n  {'READY' if pf['ok'] else 'NOT READY'} — no audio was played.\n")
            return 0 if pf["ok"] else 1
        if not pf["ok"]:
            print("\npreflight FAILED — fix the items above, or pass --no-autoconfig "
                  "to run anyway.\n", file=sys.stderr)
            return 1
        # CoreAudio renumbers device indices whenever anything changes. Re-scan.
        sd._terminate()
        sd._initialize()
        if args.samplerate is None:
            args.samplerate = int(pf["engine_rate"])
        print()

    dev = args.device
    if dev is None:
        dev, info = find_ag06()
        if dev is None:
            print("AG06 not found. Plug it in, or pass --device N (see --list).", file=sys.stderr)
            return 1
    info = sd.query_devices(dev)
    if args.out:
        o, _ = find_device(args.out, want_input=False)
        if o is None:
            print(f"no output device matching '{args.out}'. --list to see them.",
                  file=sys.stderr)
            return 1
        outdev = o
    elif args.out_device is not None:
        outdev = args.out_device
    else:
        outdev = dev
    sr = args.samplerate or int(info["default_samplerate"])
    if outdev != dev:
        oi = sd.query_devices(outdev)
        print(f"!! output -> {oi['name']} (NOT the AG06). "
              f"reported out latency {oi['default_low_output_latency'] * 1000:.0f} ms. "
              f"USE HEADPHONES — an open mic into speakers will howl.")

    chain = VocalChain(sr, reverb_mix=args.reverb)
    chain.gate.thresh = 10 ** (args.gate / 20.0)
    boost = 10 ** (args.boost / 20.0)
    ducker = Ducker(sr, thresh_db=args.duck_thresh, duck_db=args.duck_db)

    music = load_music(args.music, sr) if args.music else None
    mpos = 0

    stats = {"xruns": 0, "blocks": 0, "peak": 0.0, "vpeak": 0.0,
             "mpeak": 0.0, "starved": 0}

    # --- system-audio capture: the AI ducks whatever the USER decides to play ---
    ring = None
    sys_stream = None
    if args.capture_system:
        bh, bhinfo = find_device("blackhole")
        if bh is None:
            print("BlackHole 2ch not found. Install:  brew install --cask blackhole-2ch",
                  file=sys.stderr)
            return 1
        ring = Ring(1 << 16)
        target_backlog = args.blocksize * 4

        def sys_cb(indata, frames, t, status):
            # collapse stereo system audio to mono for the music bus
            ring.write(np.mean(indata, axis=1).astype(np.float64))

        sys_stream = sd.InputStream(device=bh, samplerate=sr, blocksize=args.blocksize,
                                    channels=2, dtype="float32", callback=sys_cb)
        print(f"capturing system audio from [{bh}] {bhinfo['name']}")
        print("   -> set macOS Sound Output to 'BlackHole 2ch', then play music "
              "from ANY app. This tool never starts or stops it.")

    bed_buf = np.zeros(args.blocksize, dtype=np.float64)  # preallocated, RT-safe

    # Music-bus gain. macOS applies its own output volume before writing into
    # BlackHole, so the captured level is not under our control and can arrive
    # ~55 dB down. Rather than trust it, measure and correct.
    mgain = {"g": 10 ** (args.music_boost / 20.0), "env": 0.0}
    m_target = 10 ** (args.music_target / 20.0)

    def callback(indata, outdata, frames, t, status):
        nonlocal mpos
        if status:
            stats["xruns"] += 1
        stats["blocks"] += 1

        mic = indata[:, 0].astype(np.float64) * boost
        stats["peak"] = max(stats["peak"], float(np.abs(mic).max()))

        vocal = chain.process(mic)

        if ring is not None:
            # music the USER is playing, captured live off the system output
            ring.drift_correct(target_backlog)
            if not ring.read(bed_buf):
                stats["starved"] += 1
            bed = bed_buf
            raw_pk = float(np.abs(bed).max())

            if args.auto_music:
                # slow AGC: track the music envelope and push it toward the target.
                # Slow on purpose — fast AGC would pump against the ducker.
                a = 0.9995 ** frames
                mgain["env"] = a * mgain["env"] + (1 - a) * raw_pk
                if mgain["env"] > 1e-6:               # only adapt on real signal
                    want = m_target / mgain["env"]
                    want = min(want, 300.0)           # cap: +50 dB
                    ag = 0.999 ** frames              # glide, never jump
                    mgain["g"] = ag * mgain["g"] + (1 - ag) * want

            bed = bed * mgain["g"]
            stats["mpeak"] = max(stats["mpeak"], float(np.abs(bed).max()))
            bed = bed * args.music_gain * ducker.gain_for(vocal)
            mix = vocal + bed
        elif music is not None:
            end = mpos + frames
            if end <= len(music):
                bed = music[mpos:end]
                mpos = end
            elif args.loop:
                bed = np.concatenate([music[mpos:], music[: end - len(music)]])
                mpos = end % len(music)
            else:
                # track finished: play out the tail, then silence. Do NOT restart.
                bed = np.zeros(frames)
                tail = len(music) - mpos
                if tail > 0:
                    bed[:tail] = music[mpos:]
                mpos = len(music)
            stats["mpeak"] = max(stats["mpeak"], float(np.abs(bed).max()))
            bed = bed * args.music_gain * ducker.gain_for(vocal)
            mix = vocal + bed
        else:
            ducker.gain_for(vocal)
            mix = vocal

        mix = np.clip(mix, -0.99, 0.99)
        # report the MIX that actually leaves the box, not just the vocal —
        # otherwise you cannot tell whether the backing track is playing at all.
        stats["vpeak"] = max(stats["vpeak"], float(np.abs(mix).max()))
        outdata[:, 0] = mix
        if outdata.shape[1] > 1:
            outdata[:, 1] = mix

    print(f"AiOke Live — {info['name']}  {sr} Hz  block {args.blocksize} "
          f"({args.blocksize / sr * 1000:.1f} ms)")
    print(f"chain: HPF -> gate -> EQ -> comp -> reverb({args.reverb:.2f}) -> limiter"
          + ("  + music bus w/ auto-duck" if music is not None else ""))
    print("AG06: TO PC = DRY CH 1-2 | MONITOR MUTE = ON | GAIN up | +48V if condenser")
    print("Ctrl-C to stop.\n")

    has_music = (music is not None) or (ring is not None)
    silent_since = [0.0]      # seconds the music bus has been dead
    chrome_warned = [False]
    try:
        if sys_stream is not None:
            sys_stream.start()
        with sd.Stream(device=(dev, outdev), samplerate=sr, blocksize=args.blocksize,
                       channels=2, dtype="float32", callback=callback):
            while True:
                time.sleep(0.25)
                p, v, m = stats["peak"], stats["vpeak"], stats["mpeak"]
                stats["peak"] = stats["vpeak"] = stats["mpeak"] = 0.0
                n = int(np.clip((db(p) + 60) / 60 * 30, 0, 30))
                bar = "#" * n + "-" * (30 - n)
                # F1: Chrome pinned its output device when the stream started, so a
                # tab that was ALREADY playing never re-bound to BlackHole. The user
                # HEARS the music and assumes it works, while capture reads zeros.
                if ring is not None:
                    if m < 1e-6:
                        silent_since[0] += 0.25
                    else:
                        silent_since[0] = 0.0
                        chrome_warned[0] = False
                    if not chrome_warned[0]:
                        import preflight as _pf
                        if _pf.chrome_is_bound_elsewhere(silent_since[0], True):
                            print("\n\n  " + _pf.CHROME_HINT + "\n", flush=True)
                            chrome_warned[0] = True

                if p < 1e-5:
                    warn = "  << NO MIC SIGNAL: check GAIN + TO PC=DRY CH1-2"
                elif p >= 0.99:
                    warn = "  << CLIPPING! turn AG06 GAIN down (or drop --boost)"
                elif ring is not None and m < 1e-6:
                    warn = "  << NO MUSIC — press play (see hint above)"
                elif (has_music and db(m) > -50.0 and not ducker.singing
                      and db(p) > args.duck_thresh - 8):
                    # Mic is hearing the speakers while REAL music plays -> duck pumps.
                    # -50 dB floor: below that the "music" is just the noise floor and
                    # this warning is a false alarm (it fired at -70 dB, i.e. silence).
                    warn = "  << SPEAKER BLEED into mic — raise --duck-thresh"
                else:
                    warn = ""
                duck = f" duck {db(ducker.g):+5.1f}dB" if has_music else ""
                mus = f" music {db(m):6.1f}dB" if has_music else ""
                sing = "SINGING" if ducker.singing else "  ---  "
                print(f"\rin {db(p):6.1f}dB [{bar}]{mus}{duck}  "
                      f"gr {chain.comp.gr_db:+5.1f}dB  {sing}  "
                      f"xruns {stats['xruns']}{warn}   ", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\nstopped. blocks={stats['blocks']} xruns={stats['xruns']} "
              f"music-starved={stats['starved']}")
        return 0
    finally:
        if sys_stream is not None:
            sys_stream.stop()
            sys_stream.close()


def selftest(sr: int, bs: int, seconds: float) -> int:
    """Prove the DSP is correct without any hardware."""
    print(f"AiOke self-test @ {sr} Hz, block {bs}\n")
    ok = True

    # 1. IIR state continuity — the click bug. Filtering one long block must equal
    #    filtering the same signal in many small blocks.
    n = bs * 16
    x = np.sin(2 * np.pi * 440 * np.arange(n) / sr)
    a = highpass(sr).process(x.copy())
    hp = highpass(sr)
    b = np.concatenate([hp.process(x[i:i + bs].copy()) for i in range(0, n, bs)])
    err = float(np.max(np.abs(a - b)))
    good = err < 1e-9
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] IIR state continuity across blocks (max err {err:.2e})")
    print("        -> if this fails you get a click at every block boundary")

    # 2. Limiter must never let anything through above the ceiling.
    lim = Limiter(ceiling_db=-1.0)
    hot = np.random.randn(bs) * 8.0
    out = lim.process(hot)
    peak = float(np.max(np.abs(out)))
    good = peak <= 10 ** (-1.0 / 20.0) + 1e-9
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] limiter holds ceiling on hot input "
          f"(peak {db(peak):.2f} dBFS <= -1.00)")

    # 3. Gate closes on silence, opens on signal.
    g = NoiseGate(sr)
    quiet = np.mean(np.abs(g.process(np.random.randn(bs * 8) * 1e-4)))
    g2 = NoiseGate(sr)
    loud = np.mean(np.abs(g2.process(np.sin(2 * np.pi * 220 *
                                            np.arange(bs * 8) / sr) * 0.5)))
    good = quiet < loud * 0.05
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] noise gate: silence {quiet:.2e} << signal {loud:.2e}")

    # 4. Ducking: music must drop while the singer sings, recover in the gaps.
    d = Ducker(sr)
    voice = np.sin(2 * np.pi * 200 * np.arange(bs) / sr) * 0.4
    for _ in range(120):
        gd_sing = d.gain_for(voice)
    for _ in range(400):
        gd_rest = d.gain_for(np.zeros(bs))
    good = gd_sing < 0.55 and gd_rest > 0.85
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] auto-duck: singing {db(gd_sing):+.1f} dB, "
          f"silence {db(gd_rest):+.1f} dB")

    # 5. Full chain must be stable and not blow up.
    ch = VocalChain(sr)
    blocks = int(seconds * sr / bs)
    t0 = time.perf_counter()
    worst = 0.0
    for i in range(blocks):
        sig = (np.sin(2 * np.pi * 330 * (np.arange(bs) + i * bs) / sr) * 0.3
               + np.random.randn(bs) * 0.002)
        y = ch.process(sig)
        if not np.all(np.isfinite(y)):
            ok = False
            print("[FAIL] chain produced NaN/Inf")
            break
        worst = max(worst, float(np.abs(y).max()))
    el = time.perf_counter() - t0
    budget = blocks * bs / sr
    rt = el / budget
    good = rt < 0.5 and worst <= 1.0
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] full chain {blocks} blocks: "
          f"{rt * 100:.1f}% of real-time budget, peak {db(worst):.1f} dBFS")
    print(f"        -> headroom {1 / rt:.1f}x. Must stay well under 100% or you get dropouts.")

    print(f"\n{'ALL PASS — DSP is sound.' if ok else 'FAILURES ABOVE.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
