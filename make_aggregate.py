"""Create a CoreAudio Aggregate Device: AG06 (mic) + a speaker, as ONE device.

WHY. Measured, on this machine:

    AG06 in -> AG06 out      : in 12.0 ms   round-trip 32.5 ms
    AG06 in -> MacBook out   : in 58.5 ms   round-trip 93.5 ms

Same input device. 5x the input latency. The only difference is that the output
lives on a DIFFERENT device, so CoreAudio has to bridge two independent clocks —
and it buys safety with enormous buffering.

An aggregate device makes them ONE device with one clock (master) and hardware
drift compensation on the other. If the theory is right, the input latency should
collapse back toward 12 ms and we get usable latency on the built-in speakers —
with no cable, no headphones, no new hardware.

This is a falsifiable claim. The script measures before/after and prints both.
Silent: writes zeros, plays nothing.
"""
import sys
import time

import objc
from CoreFoundation import (CFUUIDCreate, CFUUIDCreateString)
from Foundation import NSMutableArray, NSMutableDictionary

import numpy as np
import sounddevice as sd

sys.path.insert(0, "/Users/nguythe/aioke")
import coreaudio_cfg as ca

# --- CoreAudio symbols we need, loaded via the objc bridge -------------------
_ca_bundle = objc.loadBundle(
    "CoreAudio", globals(),
    bundle_path="/System/Library/Frameworks/CoreAudio.framework",
)
objc.loadBundleFunctions(
    _ca_bundle, globals(),
    [("AudioHardwareCreateAggregateDevice", b"i@o^I"),
     ("AudioHardwareDestroyAggregateDevice", b"iI")],
)

AGG_NAME = "AiOke (AG06 + speakers)"
AGG_UID = "com.aican.aioke.aggregate"


def device_uid(dev_id):
    import struct
    import ctypes
    raw = ca._get(dev_id, ca._fc("uid "))
    if not raw:
        return None
    ptr = struct.unpack("<Q", raw[:8])[0]
    b = ctypes.create_string_buffer(512)
    if ca._cf.CFStringGetCString(ctypes.c_void_p(ptr), b, 512, 0x08000100):
        return b.value.decode()
    return None


def destroy_existing():
    for d in ca.device_ids():
        if ca.name_of(d) == AGG_NAME:
            AudioHardwareDestroyAggregateDevice(d)
            time.sleep(0.5)
            return True
    return False


def create(master_sub_uid, speaker_uid):
    subs = NSMutableArray.array()
    for uid, drift in ((master_sub_uid, 0), (speaker_uid, 1)):
        s = NSMutableDictionary.dictionary()
        s["uid"] = uid
        s["drift"] = drift          # drift-compensate the NON-master device
        subs.append(s)

    desc = NSMutableDictionary.dictionary()
    desc["name"] = AGG_NAME
    desc["uid"] = AGG_UID
    desc["subdevices"] = subs
    desc["master"] = master_sub_uid  # AG06 is the clock master
    desc["private"] = 0
    desc["stacked"] = 0

    err, dev = AudioHardwareCreateAggregateDevice(desc, None)
    if err != 0:
        print(f"  AudioHardwareCreateAggregateDevice failed: err={err}")
        return None
    return dev


def measure(in_sub, out_sub, blocksize=128, sr=44100, latency="low"):
    """Silent duplex measurement. Writes zeros."""
    def find(sub, want_out):
        for i, d in enumerate(sd.query_devices()):
            ch = d["max_output_channels"] if want_out else d["max_input_channels"]
            if sub.lower() in d["name"].lower() and ch > 0:
                return i
        return None

    i_dev = find(in_sub, False)
    o_dev = find(out_sub, True)
    if i_dev is None or o_dev is None:
        return None
    xr = [0]

    def cb(ind, outd, frames, t, status):
        if status:
            xr[0] += 1
        outd[:] = 0.0                      # SILENT

    try:
        with sd.Stream(device=(i_dev, o_dev), samplerate=sr, blocksize=blocksize,
                       channels=2, dtype="float32", latency=latency,
                       callback=cb) as st:
            sd.sleep(2000)
            lin, lout = st.latency
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}"}
    blk = blocksize / sr * 1000
    return {"in": lin * 1000, "out": lout * 1000,
            "rtt": lin * 1000 + lout * 1000 + 2 * blk, "xruns": xr[0]}


if __name__ == "__main__":
    print("BEFORE — two separate devices (CoreAudio bridges two clocks)")
    b = measure("AG06", "MacBook Pro Speakers")
    if b and "err" not in b:
        print(f"  AG06 in -> MacBook out : in {b['in']:.1f}ms  out {b['out']:.1f}ms  "
              f"ROUND-TRIP {b['rtt']:.1f}ms  xruns {b['xruns']}")
    
    ag = ca.find_input("ag06")
    spk = ca.find_output("macbook pro speakers")
    ag_uid, spk_uid = device_uid(ag), device_uid(spk)
    print(f"\nAG06 uid     = {ag_uid}")
    print(f"speaker uid  = {spk_uid}")
    
    destroy_existing()
    dev = create(ag_uid, spk_uid)
    if dev is None:
        raise SystemExit(1)
    time.sleep(1.2)
    sd._terminate()
    sd._initialize()
    print(f"\ncreated aggregate: '{AGG_NAME}'  (AG06 = clock master, speakers drift-compensated)")
    
    for i, d in enumerate(sd.query_devices()):
        if AGG_NAME in d["name"]:
            print(f"  [{i}] in={d['max_input_channels']} out={d['max_output_channels']} "
                  f"sr={int(d['default_samplerate'])}")
    
    print("\nAFTER — one aggregate device (one clock)")
    a = measure(AGG_NAME, AGG_NAME)
    if a and "err" not in a:
        print(f"  aggregate duplex       : in {a['in']:.1f}ms  out {a['out']:.1f}ms  "
              f"ROUND-TRIP {a['rtt']:.1f}ms  xruns {a['xruns']}")
        if b and "err" not in b:
            d_ms = b["rtt"] - a["rtt"]
            print(f"\n  >>> {d_ms:+.1f} ms  "
                  f"({'AGGREGATE WINS — no cable needed' if d_ms > 5 else 'NO REAL GAIN — theory refuted'})")
    else:
        print(f"  {a}")
