"""Индикатор уровня входа: проверка, что микрофон даёт сигнал.

Запуск:  conda run -n syn --no-capture-output python tools/level_meter.py [секунды]
"""
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48_000
BLOCK = 12_000  # 250 мс на блок: 4 строки в секунду


def dbfs(x: np.ndarray) -> float:
    # float32 из sounddevice уже нормирован на [-1, 1], так что полная шкала = 1.0
    return 20 * np.log10(max(float(x), 1e-10))


def callback(indata, frames, time_info, status):
    # Вызывается из потока звуковой карты на каждый блок.
    # Здесь нельзя делать ничего медленного: если не успеть до следующего
    # блока, будет переполнение буфера (status.input_overflow).
    if status:
        print(status, file=sys.stderr)
    rms = np.sqrt(np.mean(indata**2, axis=0))  # по каналам
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
    print(f"Устройство: {dev['name']}, {SR} Гц, блок {BLOCK} сэмплов ({BLOCK / SR * 1000:.0f} мс)")
    with sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=2,
                        dtype="float32", callback=callback):
        time.sleep(seconds)
