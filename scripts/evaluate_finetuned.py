
import os
import json
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from jiwer import wer
from tqdm import tqdm

ORIGINAL_MODEL = "openai/whisper-medium"
FINETUNED_MODEL = "/scratch/izar/philip/asr_project/whisper_soccer_finetuned/final"
HF_CACHE = "/scratch/izar/philip/hf_cache"
AUDIO_DIR = "/scratch/izar/philip/soccernet_audio"
GT_DIR = "/scratch/izar/philip/asr_project/goal_data/commentaries"

# Test games (same ones we held out during training — first 15 min only)
TEST_GAMES = [
    ("england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City", ["1", "2"]),
    ("england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea", ["1", "2"]),
    ("europe_uefa-champions-league/2014-2015/2015-05-06 - 21-45 Barcelona 3 - 0 Bayern Munich", ["1", "2"]),
    ("france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG", ["1", "2"]),
]
TEST_MAX_OFFSET = 900  # First 15 minutes only

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = WhisperProcessor.from_pretrained(ORIGINAL_MODEL, cache_dir=HF_CACHE)
normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

def load_test_segments():
    """Load all test audio segments and references."""
    segments = []
    for game_folder, halves in TEST_GAMES:
        for half in halves:
            audio_path = os.path.join(AUDIO_DIR, game_folder, f"{half}_224p.wav")
            gt_filename = game_folder.replace("/", "__").replace(" ", "_") + f"_{half}.json"
            gt_path = os.path.join(GT_DIR, gt_filename)

            if not os.path.exists(audio_path) or not os.path.exists(gt_path):
                print(f"MISSING: {audio_path} or {gt_path}")
                continue

            audio, sr = librosa.load(audio_path, sr=16000)
            with open(gt_path) as f:
                gt_data = json.load(f)

            for seg in gt_data:
                if seg["offset"] > TEST_MAX_OFFSET:
                    continue
                if seg["duration"] < 0.5 or seg["duration"] > 30:
                    continue
                start_sample = int(seg["offset"] * sr)
                end_sample = int((seg["offset"] + seg["duration"]) * sr)
                audio_slice = audio[start_sample:end_sample]
                if len(audio_slice) == 0:
                    continue
                ref = normalizer(seg["commentary"])
                if len(ref.strip()) == 0:
                    continue
                segments.append({"audio": audio_slice, "reference": ref})

    print(f"Loaded {len(segments)} test segments")
    return segments

def evaluate_model(model_path, segments, label):
    """Run a model on all segments and compute WER."""
    print(f"\n--- Evaluating: {label} ---")
    model = WhisperForConditionalGeneration.from_pretrained(model_path, cache_dir=HF_CACHE).to(device)
    model.eval()

    predictions = []
    references = []

    for seg in tqdm(segments):
        inputs = processor(seg["audio"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
        pred = normalizer(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
        predictions.append(pred)
        references.append(seg["reference"])

    error_rate = wer(references, predictions) * 100
    print(f"{label} WER: {error_rate:.2f}%")
    return error_rate, predictions, references

# Load test data
segments = load_test_segments()

# Evaluate both models
original_wer, _, _ = evaluate_model(ORIGINAL_MODEL, segments, "ORIGINAL (zero-shot)")
finetuned_wer, preds, refs = evaluate_model(FINETUNED_MODEL, segments, "FINE-TUNED")

# Summary
print(f"\n{'='*60}")
print(f"RESULTS COMPARISON")
print(f"{'='*60}")
print(f"Test set: 4 games x first 15 min (554 segments)")
print(f"Original (zero-shot):  {original_wer:.2f}% WER")
print(f"Fine-tuned:            {finetuned_wer:.2f}% WER")
print(f"Improvement:           {original_wer - finetuned_wer:.2f}% absolute")
print(f"{'='*60}")

# Sample predictions
print("\n--- Sample comparisons ---")
for i in range(min(5, len(preds))):
    print(f"\n[Segment {i}]")
    print(f"REF:  {refs[i][:200]}")
    print(f"PRED: {preds[i][:200]}")
