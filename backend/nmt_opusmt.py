# backend/nmt_opusmt.py
from transformers import pipeline

# Load translation pipeline (download happens once)
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

def translate_en_to_hi(text: str) -> str:
    out = translator(text, max_length=512)
    return out[0]["translation_text"]

if __name__ == "__main__":
    print(translate_en_to_hi("How are you doing today? This is a test."))
