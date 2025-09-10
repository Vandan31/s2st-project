from TTS.api import TTS
import torch
import os

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Single-speaker model (change if you want another single-speaker model)
model_name = "tts_models/multilingual/multi-dataset/mai_tts"  # single-speaker
tts_model = TTS(model_name, progress_bar=False, gpu=(device=="cuda"))

def synthesize_coqui(text: str, output_path: str = "output.wav") -> str:
    """
    Synthesize speech from text using Coqui TTS (single-speaker).

    Args:
        text (str): Input text (e.g., Hindi text)
        output_path (str): Path to save the generated WAV file

    Returns:
        str: Path to the generated audio file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[Coqui TTS] Synthesizing to: {output_path}")
    tts_model.tts_to_file(text=text, file_path=output_path)
    return output_path
