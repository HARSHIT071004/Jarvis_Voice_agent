"""CLI: list audio devices (used to set AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE)."""

from __future__ import annotations

import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    print("Input devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " (default)" if i == default_in else ""
            print(f"  [{i}] {d['name']}{marker}")
    print("Output devices:")
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            marker = " (default)" if i == default_out else ""
            print(f"  [{i}] {d['name']}{marker}")


if __name__ == "__main__":
    main()
