import os
import json
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from jiwer import wer
from tqdm import tqdm
import re

# ===== CONFIG =====
MODEL_NAME = "openai/whisper-medium"

# Liverpool 3-0 Man City - both halves
GAME_FOLDER = "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City"

AUDIO_DIR = "/scratch/izar/philip/soccernet_audio"
GT_DIR = "/scratch/izar/philip/asr_project/goal_data/commentaries"

HALVES = ["1", "2"]
N_SEGMENTS = None   # per half for first test; set to None for all segments
# ==================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

def strip_entity_tags(text):
    # Remove [team]...[team] and [player]...[player] tags from GOAL commentaries
    # The 'commentary' field doesn't have these, but be safe
    return re.sub(r'\[/?(team|player|referee|coach)\]', '', text)

predictions, references = [], []

for half in HALVES:
    print(f"\n--- Processing half {half} ---")

    # Load audio
    audio_path = os.path.join(AUDIO_DIR, GAME_FOLDER, f"{half}_224p.wav")
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio duration: {len(audio)/sr:.0f}s ({len(audio)/sr/60:.1f} min)")

    # Load ground truth
    gt_filename = GAME_FOLDER.replace("/", "__").replace(" ", "_") + f"_{half}.json"
    gt_path = os.path.join(GT_DIR, gt_filename)
    print(f"Loading ground truth: {gt_path}")
    with open(gt_path) as f:
        segments = json.load(f)
    print(f"Total segments in half {half}: {len(segments)}")

    # Process each segment
    total = len(segments) if N_SEGMENTS is None else min(N_SEGMENTS, len(segments))

    for seg in tqdm(segments[:total]):
        start = seg["offset"]
        duration = seg["duration"]

        # Skip segments that are too short or too long for Whisper
        if duration < 0.5 or duration > 30:
            continue

        # Extract audio slice
        start_sample = int(start * sr)
        end_sample = int((start + duration) * sr)
        audio_slice = audio[start_sample:end_sample]

        if len(audio_slice) == 0:
            continue

        # Run Whisper
        inputs = processor(audio_slice, sampling_rate=sr, return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Normalize both
        pred_norm = normalizer(transcription)
        ref_norm = normalizer(strip_entity_tags(seg["commentary"]))

        if len(ref_norm.strip()) == 0:
            continue

        predictions.append(pred_norm)
        references.append(ref_norm)

# Compute WER
error_rate = wer(references, predictions)
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Model:          {MODEL_NAME}")
print(f"Dataset:        GOAL (human-verified SoccerNet transcriptions)")
print(f"Game:           Liverpool 3-0 Manchester City (both halves)")
print(f"Segments scored: {len(predictions)}")
print(f"WER:            {error_rate*100:.2f}%")
print(f"{'='*60}")

# Save a few examples so we can inspect quality
print("\n--- Sample predictions vs references ---")
for i in range(min(3, len(predictions))):
    print(f"\n[Segment {i}]")
    print(f"REF:  {references[i][:200]}")
    print(f"PRED: {predictions[i][:200]}")
