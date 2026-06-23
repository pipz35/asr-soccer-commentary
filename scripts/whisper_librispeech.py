import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper medium
print("Loading Whisper medium...")
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()
normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
# Force English transcription (LibriSpeech is English)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# Load LibriSpeech test-clean
print("Loading LibriSpeech test-clean...")
ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
print(f"Loaded {len(ds)} samples")

# Run inference
predictions, references = [], []

# Start with a subset to verify it works, then scale up
N = 100  # change to len(ds) for full test set
for i in tqdm(range(N)):
    sample = ds[i]
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    predictions.append(normalizer(transcription))
    references.append(normalizer(sample["text"]))

# Compute WER
error_rate = wer(references, predictions)
print(f"\nWER on {N} samples: {error_rate*100:.2f}%")

