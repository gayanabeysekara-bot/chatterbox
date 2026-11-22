"""Clone your own voice with Chatterbox-Nano (CPU-friendly) or Turbo (GPU).

Usage:
  python clone_my_voice.py my_voice.wav script.txt narration.wav
  python clone_my_voice.py my_voice.wav "Hello, this is my cloned voice." out.wav

my_voice.wav: 10-30 s of your own clean speech (quiet room, no music).
"""
import os
import sys

import torch
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ref, text_or_file, out = sys.argv[1:]
    text = open(text_or_file, encoding="utf-8").read() if os.path.isfile(text_or_file) else text_or_file

    if torch.cuda.is_available():
        device, nano = "cuda", False   # Turbo: better quality on GPU
    elif torch.backends.mps.is_available():
        device, nano = "mps", False
    else:
        device, nano = "cpu", True     # Nano: fast on CPU

    print(f"Loading {'Nano' if nano else 'Turbo'} on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device, nano=nano)

    # Generate paragraph by paragraph so long scripts stay stable.
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    silence = torch.zeros(1, int(model.sr * 0.4))
    parts = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk[:60]}...")
        parts += [model.generate(chunk, audio_prompt_path=ref), silence]

    ta.save(out, torch.cat(parts, dim=1), model.sr)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
