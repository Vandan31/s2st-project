from backend.asr.asr_whisper import transcribe
from backend.nmt_opusmt import translate_en_to_hi
from backend.tts.tts_speak import synthesize_espeak
from pydub import AudioSegment
import os
from pathlib import Path

def normalize_audio(in_path: str, out_path: str):
    """Convert input audio to mono 16kHz WAV."""
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

def run_pipeline(in_audio: str, out_dir: str = "data/output"):
    """Full pipeline: ASR → NMT → TTS"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Normalize input
    normalized = os.path.join(out_dir, "normalized.wav")
    normalize_audio(in_audio, normalized)

    # Run ASR (English)
    en_text = transcribe(normalized)

    # Translate (EN → HI)
    hi_text = translate_en_to_hi(en_text)

    # Run TTS (Hindi)
    tts_wav = os.path.join(out_dir, "tts_hi.wav")
    synthesize_espeak(hi_text, tts_wav)

    return {"en_text": en_text, "hi_text": hi_text, "tts_wav": tts_wav}

if __name__ == "__main__":
    result = run_pipeline("data/tests/sample.wav", out_dir="data/output")
    print(result)
