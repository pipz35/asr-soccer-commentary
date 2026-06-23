import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm

# ===== CONFIG =====
MODEL_NAME = "openai/whisper-medium"
DATASET_NAME = "distil-whisper/tedlium-long-form"
DATASET_CONFIG = "all"
SPLIT = "test"
N_SAMPLES = 100  # change to None to run full test set
# ==================

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print(f"Loading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# Load dataset
print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}) - {SPLIT} split...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
print(f"Loaded {len(ds)} samples")

# Filter out empty or very short transcriptions
valid_indices = [
    i for i in range(len(ds))
    if ds[i]["text"] is not None
    and len(ds[i]["text"].strip()) > 0
    and ds[i]["text"].strip().lower() != "ignore_time_segment_in_scoring"
]
print(f"Valid samples (after filtering): {len(valid_indices)}")

# Determine how many to run
total = len(valid_indices) if N_SAMPLES is None else min(N_SAMPLES, len(valid_indices))

# Run inference
predictions, references = [], []

for idx in tqdm(range(total)):
    sample = ds[valid_indices[idx]]
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    # Resample if needed (Whisper expects 16kHz)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Skip very long audio (>30s) to avoid Whisper truncation issues
    duration = len(audio) / sr
    if duration > 30:
        # Split into 30s chunk (just use first 30s for now)
        audio = audio[:30 * sr]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    pred_norm = normalizer(transcription)
    ref_norm = normalizer(sample["text"])

    # Skip if normalization produces empty string
    if len(ref_norm.strip()) == 0:
        continue

    predictions.append(pred_norm)
    references.append(ref_norm)

# Results
error_rate = wer(references, predictions)
print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Model:          {MODEL_NAME}")
print(f"Dataset:        {DATASET_NAME} ({DATASET_CONFIG})")
print(f"Split:          {SPLIT}")
print(f"Samples scored: {len(predictions)}")
print(f"WER:            {error_rate*100:.2f}%")
print(f"{'='*50}")
