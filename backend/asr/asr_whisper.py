# backend/asr_whisper.py
import whisper

# Load once (heavy) - small model recommended for CPU
model = whisper.load_model("small")  # options: tiny, base, small, medium, large

def transcribe(path: str) -> str:
    """
    Transcribe audio file at path -> returns English transcript (string).
    """
    # whisper returns dict with keys like "text"
    result = model.transcribe(path, language="en")
    text = result.get("text", "").strip()
    return text

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/tests/sample.wav"
    print("Transcribing:", p)
    print(transcribe(p))
