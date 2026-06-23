
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_NAME = "openai/whisper-medium"
TRAIN_DATASET = "/scratch/izar/philip/asr_project/soccer_finetune_dataset/train_dataset"
TEST_DATASET = "/scratch/izar/philip/asr_project/soccer_finetune_dataset/test_dataset"
OUTPUT_DIR = "/scratch/izar/philip/asr_project/whisper_soccer_finetuned"
HF_CACHE = "/scratch/izar/philip/hf_cache"

print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.gradient_checkpointing_enable()

print("Loading datasets...")
train_ds = load_from_disk(TRAIN_DATASET)
test_ds = load_from_disk(TEST_DATASET)
print(f"Train: {len(train_ds)} samples")
print(f"Test:  {len(test_ds)} samples")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("Preprocessing train set...")
train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=1)
print("Preprocessing test set...")
test_ds = test_ds.map(prepare_dataset, remove_columns=test_ds.column_names, num_proc=1)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=200,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,
    report_to=[],
    push_to_hub=False,
    dataloader_num_workers=4,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Starting training...")
print(f"  Model: {MODEL_NAME}")
print(f"  Effective batch size: {4 * 4}")
print(f"  Learning rate: 1e-5")
print(f"  Max steps: 2000")
trainer.train()

trainer.save_model(OUTPUT_DIR + "/final")
processor.save_pretrained(OUTPUT_DIR + "/final")
print(f"\nDone! Model saved to: {OUTPUT_DIR}/final")
