from jiwer import wer, cer
from backend.asr.asr_whisper import transcribe
import glob, csv

test_files = glob.glob("data/tests/*.wav")
with open("experiments/wer_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    # Add header with model name and dataset
    w.writerow(["ASR Model", "Test Dataset", "Audio", "Ref", "Hyp", "WER (%)", "CER (%)"])
    for p in test_files:
        ref_path = p.replace(".wav", ".txt")
        try:
            with open(ref_path, "r") as rf:
                ref = rf.read().strip()
        except FileNotFoundError:
            print(f"Reference file not found for {p}, skipping")
            continue

        hyp = transcribe(p)
        w_val = wer(ref, hyp) * 100  # convert to %
        c_val = cer(ref, hyp) * 100  # convert to %
        w.writerow(["Whisper", "Common Voice Subset", p, ref, hyp, round(w_val, 2), round(c_val, 2)])

print("Done -> experiments/wer_results.csv")
