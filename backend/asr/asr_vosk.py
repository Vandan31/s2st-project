import wave
import json
import sys
from vosk import Model, KaldiRecognizer
from pathlib import Path

MODEL_DIR = Path("data/models/vosk/en-us")
if not MODEL_DIR.exists():
    raise RuntimeError(f"Vosk model not found at {MODEL_DIR}, please download it.")

model = Model(str(MODEL_DIR))


def transcribe(wav_path: str) -> str:
    """Transcribe speech to text using Vosk"""
    wf = wave.open(wav_path, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        raise ValueError("Audio file must be WAV mono PCM at 8kHz or 16kHz.")

    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))

    final = json.loads(rec.FinalResult())
    results.append(final)

    text = " ".join([r.get("text", "") for r in results]).strip()
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.asr.asr_vosk <wav_file>")
        sys.exit(1)

    wav_file = sys.argv[1]
    print("Transcribing:", wav_file)
    try:
        text = transcribe(wav_file)
        print("Recognized text:", text)
    except Exception as e:
        import traceback
        traceback.print_exc()
