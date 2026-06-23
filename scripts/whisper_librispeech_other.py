import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm

# ===== CONFIG =====
MODEL_NAME = "openai/whisper-medium"
DATASET_NAME = "openslr/librispeech_asr"
DATASET_CONFIG = "other"
SPLIT = "test"
N_SAMPLES = 100  # change to None for full test set
# ==================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}) - {SPLIT} split...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
print(f"Loaded {len(ds)} samples")

total = len(ds) if N_SAMPLES is None else min(N_SAMPLES, len(ds))
predictions, references = [], []

for i in tqdm(range(total)):
    sample = ds[i]
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    predictions.append(normalizer(transcription))
    references.append(normalizer(sample["text"]))

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
