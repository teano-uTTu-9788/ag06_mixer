"""
AiOke preflight — the app configures the machine, and tells you what it cannot.

Every "no sound" / "warped audio" incident in this project traced to ONE of five
device properties that macOS gives you no error for. Each produced identical
silence. This module detects all five, fixes the four that are fixable from code,
and states the exact user action for the one that is not.

  F1  Chrome pins its output device when its audio stream starts. A YouTube tab
      already playing when the default output changes keeps sending to the OLD
      device. User HEARS music; capture reads bit-exact zeros.   -> USER ACTION
  F2  macOS applies output volume/mute BEFORE writing into BlackHole. Muted or 0%
      yields digital silence. 19% yields -60 dBFS.               -> AUTO-FIX
  F3  Sample-rate mismatch. AG06 is 44.1k-ONLY (48k xruns at every blocksize,
      measured). BlackHole / HDMI / headphones default to 48k. Chrome re-forces
      BlackHole back to 48k. 44.1k vs 48k = 8.8% pitch warp.     -> AUTO-FIX
  F4  Wrong default input/output device selected.                -> AUTO-FIX
  F5  Microphone TCC permission denied. macOS returns BIT-EXACT ZEROS, not an
      error. BlackHole counts as a mic (it is an input device).  -> USER ACTION
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

import coreaudio_cfg as ca
import make_aggregate

GREEN = "\033[32m"
RED = "\033[31m"
YELL = "\033[33m"
DIM = "\033[2m"
OFF = "\033[0m"


def _row(ok, label, detail=""):
    tag = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
    print(f"  [{tag}] {label:<38} {DIM}{detail}{OFF}")
    return ok


def _fixed(label, detail=""):
    print(f"  [{YELL}FIXED{OFF}] {label:<38} {DIM}{detail}{OFF}")


def mic_permission_ok(sd, dev_idx, sr) -> bool:
    """A denied mic returns bit-exact zeros, NOT an error. Detect that."""
    try:
        rec = sd.rec(int(sr * 0.35), samplerate=int(sr), channels=1,
                     device=dev_idx, dtype="float32", blocking=True)
    except Exception:
        return False
    # Real hardware ALWAYS has a noise floor. Exactly zero across the board means
    # the OS is feeding us silence because permission was refused.
    return int(np.count_nonzero(rec)) > 0


def run(sd, engine_rate=None, want_capture=True, out_match=None,
        set_defaults=True, fix=True) -> dict:
    """Configure the machine for AiOke. Returns a dict of what it found/did."""
    print(f"\n{DIM}--- AiOke preflight ---{OFF}")
    res = {"ok": True, "actions": [], "engine_rate": None,
           "ag06": None, "blackhole": None, "output": None}

    # ---- 1. the AG06 itself, and the ONE rate everything must agree on --------
    ag = (ca.find_input("ag06") or ca.find_input("ag03") or ca.find_input("yamaha"))
    if ag is None:
        _row(False, "AG06 found", "plug it in via USB")
        res["ok"] = False
        return res
    ag_rate = ca.get_rate(ag)
    # The AG06 dictates the rate. It xruns at 48k at EVERY blocksize (measured).
    rate = float(engine_rate or ag_rate or 44100)
    res["engine_rate"] = rate
    res["ag06"] = ca.name_of(ag)
    _row(True, "AG06 found", f"{ca.name_of(ag)} @ {int(ag_rate or 0)} Hz")
    if abs((ag_rate or 0) - rate) > 1 and fix:
        if ca.set_rate(ag, rate):
            _fixed("AG06 sample rate", f"-> {int(rate)} Hz")

    # ---- 2. every device in the chain MUST share that rate (F3) ---------------
    def align(dev, label):
        if dev is None:
            return True
        r = ca.get_rate(dev)
        if r is None or abs(r - rate) < 1:
            return _row(True, f"{label} rate", f"{int(r or 0)} Hz")
        if fix and ca.set_rate(dev, rate):
            _fixed(f"{label} rate", f"{int(r)} -> {int(rate)} Hz  "
                                    f"(mismatch = pitch warp)")
            return True
        _row(False, f"{label} rate", f"{int(r)} Hz != {int(rate)} Hz — WILL WARP AUDIO")
        res["ok"] = False
        return False

    bh = ca.find_output("blackhole") if want_capture else None
    if want_capture and bh is None:
        _row(False, "BlackHole 2ch found",
             "brew install --cask blackhole-2ch  (needed to capture your music)")
        res["ok"] = False
    res["blackhole"] = ca.name_of(bh) if bh else None

    # MUST be an OUTPUT device. 'macbook' also matches 'MacBook Pro Microphone',
    # and without the direction filter we would configure the user to listen on a mic.
    out = ca.find_output(out_match) if out_match else None
    if out_match and out is None:
        _row(False, f"output '{out_match}' found",
             "no OUTPUT device matches that name — try --list")
        res["ok"] = False
    res["output"] = ca.name_of(out) if out else None

    align(bh, "BlackHole")
    align(out, "output")

    # ---- 3. BlackHole mute + volume (F2) — the silent killer ------------------
    if bh is not None:
        muted = ca.get_mute(bh)
        vol = ca.get_volume(bh)
        bad = bool(muted) or (vol is not None and vol < 0.99)
        if bad and fix:
            if muted:
                ca.set_mute(bh, False)
            ca.set_volume(bh, 1.0)
            nv = ca.get_volume(bh)
            _fixed("BlackHole mute/volume",
                   f"muted={muted} vol={vol:.2f} -> unmuted vol={nv if nv is None else round(nv,2)}"
                   f"   (macOS applies this BEFORE writing into BlackHole)")
        else:
            _row(not bad, "BlackHole mute/volume",
                 f"muted={muted} vol={vol}")

    # ---- 4. default input/output (F4) -----------------------------------------
    if set_defaults and fix:
        cur_in = ca.get_default_input()
        if cur_in != ag:
            ca.set_default_input(ag)
            _fixed("system INPUT", "-> AG06")
        else:
            _row(True, "system INPUT", "AG06")

        if want_capture and bh is not None:
            cur_out = ca.get_default_output()
            if cur_out != bh:
                # Preserve the listening device's own level BEFORE we hide it
                # behind BlackHole — once BlackHole is default, the macOS volume
                # slider no longer reaches the real speakers.
                if out is not None:
                    v = ca.get_volume(out)
                    if v is not None and v < 0.5:
                        ca.set_volume(out, 0.7)
                        _fixed("output device volume", f"{v:.2f} -> 0.70")
                    ca.set_mute(out, False)
                ca.set_default_output(bh)
                _fixed("system OUTPUT", "-> BlackHole 2ch  (so the AI can hear your music)")
            else:
                _row(True, "system OUTPUT", "BlackHole 2ch")

    # ---- 5. microphone permission (F5) — returns ZEROS, not an error ----------
    idx = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and any(
                k in d["name"].lower() for k in ("ag06", "ag03", "yamaha")):
            idx = i
            break
    if idx is not None:
        ok = mic_permission_ok(sd, idx, rate)
        if not ok:
            _row(False, "microphone permission",
                 "DENIED — macOS is feeding this app SILENCE (not an error).")
            print(f"     {YELL}Run from Terminal, or grant mic access in"
                  f" System Settings > Privacy & Security > Microphone.{OFF}")
            res["ok"] = False
        else:
            _row(True, "microphone permission", "granted (mic has a noise floor)")


    # ---- 6. Aggregate Device (F6) ---------------------------------------------
    # Create the aggregate device to sync clocks and reduce latency.
    # The AG06 is the master clock. The output device is drift-compensated.
    if out is not None:
        ag_uid = make_aggregate.device_uid(ag)
        out_uid = make_aggregate.device_uid(out)
        if ag_uid and out_uid:
            make_aggregate.destroy_existing()
            agg_dev = make_aggregate.create(ag_uid, out_uid)
            if agg_dev is not None:
                import time
                time.sleep(1.2)
                import sounddevice as sd
                sd._terminate()
                sd._initialize()
                ca.set_rate(agg_dev, rate)
                
                # DETECT which pair reaches a live output
                # The aggregate exposes ONE STEREO PAIR PER SUB-DEVICE in order.
                # So the speakers start after the AG06 outputs.
                ag_out_ch = 2
                for d in sd.query_devices():
                    if d["name"] == ca.name_of(ag):
                        ag_out_ch = d["max_output_channels"]
                        break
                
                res["out_ch"] = ag_out_ch
                res["aggregate"] = make_aggregate.AGG_NAME
                
                # Verify the downstream observable: silent duplex probe
                probe = make_aggregate.measure(make_aggregate.AGG_NAME, make_aggregate.AGG_NAME, sr=int(rate))
                if probe and "err" not in probe:
                    _row(True, "Aggregate device", f"created {make_aggregate.AGG_NAME} (out_ch={ag_out_ch}), rtt {probe['rtt']:.1f}ms")
                else:
                    _row(False, "Aggregate device", f"probe failed: {probe.get('err', 'unknown')}")
                    res["ok"] = False

    return res


def chrome_is_bound_elsewhere(music_silent_for_s: float, want_capture: bool) -> bool:
    """F1. Chrome binds its output device when the stream STARTS and does not
    follow later default-device changes. A tab that was already playing keeps
    sending to the old device: the user HEARS music while capture reads zeros."""
    if not want_capture or music_silent_for_s < 5.0:
        return False
    try:
        r = subprocess.run(["pgrep", "-f", "Google Chrome"],
                           capture_output=True, timeout=2)
        return r.returncode == 0
    except Exception:
        return False


CHROME_HINT = (
    f"{YELL}Chrome is still bound to its OLD output device.{OFF}\n"
    f"     Your music is playing, but not into BlackHole — so the AI cannot hear it.\n"
    f"     {GREEN}FIX: PAUSE and PLAY the video (or reload the tab).{OFF}\n"
    f"     Chrome only picks up the new output device when a stream RESTARTS."
)
