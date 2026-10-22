"""Record a few seconds from the default input and save as a mono WAV.

Usage:  conda run -n syn --no-capture-output python explore/record.py [seconds] [name]
Result: recordings/<name>.wav
"""
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 48_000
OUT_DIR = Path(__file__).resolve().parent.parent / "recordings"


def record(seconds: float, sr: int = SR) -> np.ndarray:
    # sd.rec starts recording in the background and returns immediately;
    # sd.wait blocks until the requested number of frames is captured.
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=2, dtype="float32")
    sd.wait()
    # The Q502USB sends the stereo main mix; a centered mic is identical in L and R,
    # so averaging loses nothing and gives one signal x[n].
    return audio.mean(axis=1)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    name = sys.argv[2] if len(sys.argv) > 2 else "test"
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Recording {seconds:.0f} s ...", flush=True)
    x = record(seconds)
    peak_db = 20 * np.log10(max(np.max(np.abs(x)), 1e-10))
    print(f"Done: {len(x)} samples, peak {peak_db:.1f} dBFS")

    path = OUT_DIR / f"{name}.wav"
    sf.write(path, x, SR)
    print(f"Saved to {path}")
