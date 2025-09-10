# backend/tts_espeak.py
import subprocess
import shlex
import os
from pathlib import Path

def synthesize_espeak(text: str, out_wav: str) -> str:
    """
    Uses espeak-ng to synthesize Hindi speech (offline).
    Returns path to generated wav.
    """
    out_dir = Path(out_wav).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # -v hi => hindi voice, -w output file
    cmd = f'espeak-ng -v hi -s 140 -w "{out_wav}" "{text}"'
    subprocess.run(shlex.split(cmd), check=True)
    return str(out_wav)

if __name__ == "__main__":
    out = synthesize_espeak("नमस्ते दुनिया, यह एक परीक्षण है।", "data/tests/espeak_test.wav")
    print("WAV:", out)
