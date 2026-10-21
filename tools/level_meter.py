"""Input level meter: check that the microphone delivers a signal.

Usage:  conda run -n syn --no-capture-output python tools/level_meter.py [seconds]
"""
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48_000
BLOCK = 12_000  # 250 ms per block: 4 lines per second


def dbfs(x: np.ndarray) -> float:
    # float32 from sounddevice is already normalized to [-1, 1], so full scale = 1.0
    return 20 * np.log10(max(float(x), 1e-10))


def callback(indata, frames, time_info, status):
    # Called from the audio thread for every block.
    # Nothing slow may happen here: if we miss the next block,
    # the buffer overflows (status.input_overflow).
    if status:
        print(status, file=sys.stderr)
    rms = np.sqrt(np.mean(indata**2, axis=0))  # per channel
    peak = np.max(np.abs(indata), axis=0)
    bars = "  ".join(
        f"ch{i}: {dbfs(r):6.1f} dBFS |{'#' * int(max(0, dbfs(r) + 90) / 2):<45}|"
        for i, r in enumerate(rms)
    )
    clip = "  CLIP!" if np.any(peak >= 0.99) else ""
    print(bars + clip, flush=True)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    dev = sd.query_devices(kind="input")
    print(f"Device: {dev['name']}, {SR} Hz, block {BLOCK} samples ({BLOCK / SR * 1000:.0f} ms)")
    with sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=2,
                        dtype="float32", callback=callback):
        time.sleep(seconds)
