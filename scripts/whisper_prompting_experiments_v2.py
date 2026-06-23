"""
Whisper Prompting Experiments for Soccer Commentary ASR
========================================================
Runs multiple prompting strategies on fine-tuned Whisper model.

Strategies:
  N  = No prompt (baseline)
  C  = Context prompt
  PL = Player list prompt (per game)
  GT = Ground truth prompt (cheating upper bound)
  RT = Random training transcript examples (15)

Metrics:
  - Global WER
  - Entity WER (entity-containing segments only)
  - Entity Detection Accuracy (binary per entity)
"""

import os
import sys
import json
import random
import argparse
import torch
import librosa
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from jiwer import wer
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
ORIGINAL_MODEL = "openai/whisper-medium"
FINETUNED_MODEL = "/scratch/izar/philip/asr_project/whisper_soccer_finetuned/final"
HF_CACHE = "/scratch/izar/philip/hf_cache"
AUDIO_DIR = "/scratch/izar/philip/soccernet_audio"
GT_DIR = "/scratch/izar/philip/asr_project/goal_data/commentaries"

TEST_MAX_OFFSET = 900  # First 15 minutes only

# Test games: (audio_path, gt_file_prefix)
# audio_path is relative to AUDIO_DIR
# gt_file_prefix matches the GOAL commentary filename (without _1.json/_2.json)
TEST_GAMES = [
    {
        "audio_path": "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City",
        "gt_prefix": "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City",
        "name": "Liverpool 3-0 Man City",
        "roster": (
            "Mignolet, Clyne, Flanagan, Lovren, Can, Henderson, Lallana, Coutinho, "
            "Firmino, Milner, Sturridge, Origi, Ibe, Benteke, Allen, Skrtel, "
            "Kolo Toure, Yaya Toure, Hart, Sagna, Otamendi, Kompany, Kolarov, "
            "Clichy, Fernandinho, Fernando, Silva, David Silva, Navas, Sterling, "
            "Aguero, Zabaleta, Iheanacho, Bony, Caballero, Shaqiri, "
            "Liverpool, Manchester City, Leicester, Klopp, Pellegrini"
        ),
    },
    {
        "audio_path": "england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea",
        "gt_prefix": "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea",
        "name": "Arsenal 3-0 Chelsea",
        "roster": (
            "Cech, Bellerin, Mustafi, Koscielny, Monreal, Coquelin, Cazorla, "
            "Walcott, Ozil, Iwobi, Sanchez, Xhaka, Elneny, Giroud, Gibbs, "
            "Oxlade Chamberlain, Lucas Perez, Courtois, Azpilicueta, David Luiz, "
            "Cahill, Alonso, Kante, Matic, Pedro, Hazard, Costa, Diego Costa, "
            "Willian, Fabregas, Moses, Batshuayi, Ivanovic, Oscar, "
            "Arsenal, Chelsea, Conte, Wenger"
        ),
    },
    {
        "audio_path": "europe_uefa-champions-league/2014-2015/2015-05-06 - 21-45 Barcelona 3 - 0 Bayern Munich",
        "gt_prefix": "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich",
        "name": "Barcelona 3-0 Bayern Munich",
        "roster": (
            "Ter Stegen, Dani Alves, Pique, Mascherano, Jordi Alba, Busquets, "
            "Rakitic, Iniesta, Messi, Suarez, Neymar, Xavi, Bartra, Neuer, "
            "Boateng, Benatia, Bernat, Xabi Alonso, Lahm, Schweinsteiger, "
            "Muller, Robben, Lewandowski, Rafinha, Alaba, Gotze, Ribery, Thiago, "
            "Barcelona, Bayern Munich, Bayern, Guardiola, Luis Enrique"
        ),
    },
    {
        "audio_path": "france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG",
        "gt_prefix": "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG",
        "name": "Marseille 2-3 PSG",
        "roster": (
            "Mandanda, Fanni, Morel, Mendy, Romao, Imbula, Lemina, Payet, "
            "Alessandrini, Ayew, Gignac, Batshuayi, Thauvin, Ocampos, Djedje, "
            "Sirigu, Van der Wiel, Marquinhos, Thiago Silva, Maxwell, Verratti, "
            "Matuidi, Motta, Rabiot, Pastore, Cavani, Lavezzi, Ibrahimovic, David Luiz, "
            "Marseille, PSG, Paris Saint Germain, Bielsa, Laurent Blanc"
        ),
    },
]

NUM_RT_EXAMPLES = 15
RANDOM_SEED = 42


# ============================================================
# Data Loading
# ============================================================

def extract_entities(entity_text):
    """Extract entities from GOAL format: [player]Sterling[player], [team]Liverpool[team], etc."""
    if not entity_text:
        return []
    entities = []
    # Pattern: [type]name[type] (same tag on both sides)
    pattern = r'\[(\w+)\](.*?)\[\1\]'
    for match in re.finditer(pattern, entity_text):
        etype = match.group(1)
        ename = match.group(2).strip()
        if ename:
            entities.append((etype, ename))
    return entities


def load_test_segments():
    """Load all test audio segments with references and entity info."""
    segments = []

    for game in TEST_GAMES:
        print(f"\nLoading: {game['name']}")
        game_loaded = 0
        game_skipped = 0

        for half in ["1", "2"]:
            gt_file = os.path.join(GT_DIR, game["gt_prefix"] + f"_{half}.json")
            if not os.path.exists(gt_file):
                print(f"  WARNING: GT file not found: {gt_file}")
                continue

            with open(gt_file, "r") as f_gt:
                annotations = json.load(f_gt)

            audio_file = os.path.join(AUDIO_DIR, game["audio_path"], f"{half}_224p.wav")
            if not os.path.exists(audio_file):
                print(f"  WARNING: Audio file not found: {audio_file}")
                continue

            for ann in annotations:
                offset = ann.get("offset", 0)
                duration = ann.get("duration", 0)
                commentary = ann.get("commentary", "")
                entity_text = ann.get("entity", "")

                # Only first 15 minutes
                if offset >= TEST_MAX_OFFSET:
                    continue

                # Skip segments > 30 seconds (Whisper limit)
                if duration > 30:
                    game_skipped += 1
                    continue

                # Skip empty
                if not commentary.strip():
                    game_skipped += 1
                    continue

                # Load audio segment
                try:
                    audio, sr = librosa.load(audio_file, sr=16000,
                                             offset=offset, duration=duration)
                    if len(audio) < 160:  # < 10ms
                        game_skipped += 1
                        continue
                except Exception as e:
                    print(f"  Audio load error at offset {offset}: {e}")
                    game_skipped += 1
                    continue

                entities = extract_entities(entity_text)

                segments.append({
                    "audio": audio,
                    "reference": commentary.strip(),
                    "entity_text": entity_text,
                    "entities": entities,
                    "game_name": game["name"],
                    "roster": game["roster"],
                    "offset": offset,
                    "duration": duration,
                })
                game_loaded += 1

        print(f"  Loaded: {game_loaded}, Skipped: {game_skipped}")

    print(f"\nTotal test segments: {len(segments)}")
    return segments


def load_training_transcripts(n=15):
    """Load random training transcripts for RT strategy."""
    random.seed(RANDOM_SEED)
    all_transcripts = []

    # All GT files that are NOT test games
    test_prefixes = {g["gt_prefix"] for g in TEST_GAMES}

    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".json"):
            continue
        # Check if this is a test file
        prefix = fname.rsplit("_", 1)[0]  # Remove _1.json or _2.json suffix
        if prefix in test_prefixes:
            continue

        fpath = os.path.join(GT_DIR, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            for ann in data:
                commentary = ann.get("commentary", "").strip()
                if commentary and len(commentary) > 20:
                    all_transcripts.append(commentary)
        except:
            continue

    print(f"Found {len(all_transcripts)} training transcripts from {20-4} training games")

    if not all_transcripts:
        print("WARNING: No training transcripts found!")
        return []

    selected = random.sample(all_transcripts, min(n, len(all_transcripts)))
    print(f"Selected {len(selected)} random transcripts for RT prompt")
    for i, t in enumerate(selected):
        print(f"  RT[{i}]: {t[:80]}...")
    return selected


# ============================================================
# Prompting Strategies
# ============================================================

def get_prompt_text(strategy, segment=None, training_transcripts=None):
    """Generate prompt text for a given strategy."""
    if strategy == "N":
        return None

    elif strategy == "C":
        return (
            "Audio recording from a football match with crowd noise. "
            "English language commentary of a soccer game."
        )

    elif strategy == "PL":
        if segment and segment.get("roster"):
            return f"Football match commentary. Players: {segment['roster']}"
        return "Audio recording from a football match with crowd noise."

    elif strategy == "GT":
        if segment:
            return segment["reference"]
        return None

    elif strategy == "RT":
        if training_transcripts:
            prompt = " ".join(training_transcripts)
            # Whisper prompt limit is 224 tokens, ~900 chars is safe
            if len(prompt) > 850:
                prompt = prompt[:850]
            return prompt
        return None

    return None


# ============================================================
# Evaluation
# ============================================================

def compute_entity_detection_accuracy(segments, predictions, normalizer):
    """Binary entity detection: is each entity found in the ASR output?

    Full match = all words of the entity found in prediction.
    Partial match = at least one word found.
    """
    full_matches = []
    partial_matches = []
    details = []

    for seg, pred in zip(segments, predictions):
        if not seg["entities"]:
            continue

        pred_norm = normalizer(pred).lower()

        for etype, ename in seg["entities"]:
            ename_norm = normalizer(ename).lower()
            words = ename_norm.split()
            if not words:
                continue

            all_found = all(w in pred_norm for w in words)
            any_found = any(w in pred_norm for w in words)

            full_matches.append(1 if all_found else 0)
            partial_matches.append(1 if any_found else 0)
            details.append({
                "type": etype, "name": ename,
                "full": all_found, "partial": any_found,
                "pred_snippet": pred[:80],
            })

    n = len(full_matches)
    if n == 0:
        return {"full": 0, "partial": 0, "n": 0, "details": details}

    return {
        "full": sum(full_matches) / n * 100,
        "partial": sum(partial_matches) / n * 100,
        "n": n,
        "details": details,
    }


def compute_entity_wer(segments, predictions, normalizer):
    """WER computed only on segments that contain entities."""
    refs = []
    preds = []

    for seg, pred in zip(segments, predictions):
        if seg["entities"]:
            r = normalizer(seg["reference"])
            p = normalizer(pred)
            if r.strip():
                refs.append(r)
                preds.append(p)

    if not refs:
        return 0.0, 0

    return wer(refs, preds) * 100, len(refs)


def run_experiment(model, processor, normalizer, segments, strategy,
                   training_transcripts=None):
    """Run one prompting strategy across all segments."""
    device = next(model.parameters()).device
    predictions = []

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    for seg in tqdm(segments, desc=f"  {strategy}"):
        inputs = processor(
            seg["audio"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        prompt_text = get_prompt_text(strategy, segment=seg,
                                      training_transcripts=training_transcripts)

        with torch.no_grad():
            if prompt_text is not None:
                prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt")
                prompt_ids = prompt_ids.to(device)
                predicted_ids = model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    prompt_ids=prompt_ids,
                )
            else:
                predicted_ids = model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids,
                )

        pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(pred)

    # Normalize
    refs_norm = [normalizer(seg["reference"]) for seg in segments]
    preds_norm = [normalizer(p) for p in predictions]

    # Filter empty refs
    valid = [(r, p) for r, p in zip(refs_norm, preds_norm) if r.strip()]
    if not valid:
        return None, predictions

    valid_refs, valid_preds = zip(*valid)

    # Global WER
    global_wer_val = wer(list(valid_refs), list(valid_preds)) * 100

    # Entity WER
    entity_wer_val, n_ent_segs = compute_entity_wer(segments, predictions, normalizer)

    # Entity detection accuracy
    ent_acc = compute_entity_detection_accuracy(segments, predictions, normalizer)

    results = {
        "strategy": strategy,
        "global_wer": global_wer_val,
        "entity_wer": entity_wer_val,
        "n_entity_segments": n_ent_segs,
        "entity_detect_full": ent_acc["full"],
        "entity_detect_partial": ent_acc["partial"],
        "n_entities": ent_acc["n"],
        "n_segments": len(segments),
    }

    return results, predictions


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["finetuned", "original"], default="finetuned")
    parser.add_argument("--strategies", type=str, default="N,C,PL,GT,RT")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    model_path = FINETUNED_MODEL if args.model == "finetuned" else ORIGINAL_MODEL

    print("=" * 70)
    print("WHISPER PROMPTING EXPERIMENTS — SOCCER COMMENTARY ASR")
    print("=" * 70)
    print(f"Model:      {args.model} ({model_path})")
    print(f"Strategies: {strategies}")
    print(f"Test set:   4 games x first 15 min (half 1 only)")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load processor + normalizer
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(ORIGINAL_MODEL, cache_dir=HF_CACHE)
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)

    # Load model
    print(f"Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, cache_dir=HF_CACHE
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")

    # Load test segments
    print("\n--- Loading test segments ---")
    segments = load_test_segments()
    if not segments:
        print("ERROR: No segments loaded!")
        sys.exit(1)

    # Count entity stats
    n_with_entities = sum(1 for s in segments if s["entities"])
    n_total_entities = sum(len(s["entities"]) for s in segments)
    print(f"Segments with entities: {n_with_entities}/{len(segments)}")
    print(f"Total entity mentions: {n_total_entities}")

    # Load training transcripts for RT
    training_transcripts = None
    if "RT" in strategies:
        print("\n--- Loading training transcripts ---")
        training_transcripts = load_training_transcripts(NUM_RT_EXAMPLES)

    # Run experiments
    all_results = []
    all_predictions = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        result, preds = run_experiment(
            model, processor, normalizer, segments, strategy,
            training_transcripts=training_transcripts
        )

        if result:
            all_results.append(result)
            all_predictions[strategy] = preds

            print(f"\n  Global WER:                {result['global_wer']:.2f}%")
            print(f"  Entity WER:                {result['entity_wer']:.2f}% ({result['n_entity_segments']} segs)")
            print(f"  Entity Detection (full):   {result['entity_detect_full']:.1f}% ({result['n_entities']} entities)")
            print(f"  Entity Detection (partial): {result['entity_detect_partial']:.1f}%")

    # Summary table
    print("\n\n" + "=" * 95)
    print("RESULTS SUMMARY")
    print("=" * 95)
    print(f"{'Strategy':<10} {'Global WER':>12} {'Entity WER':>12} {'Detect(full)':>14} {'Detect(part)':>14} {'#Ent':>6}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['strategy']:<10} {r['global_wer']:>11.2f}% {r['entity_wer']:>11.2f}% "
              f"{r['entity_detect_full']:>13.1f}% {r['entity_detect_partial']:>13.1f}% "
              f"{r['n_entities']:>6}")
    print("=" * 95)

    # Sample predictions
    print("\n\nSAMPLE PREDICTIONS (first 5 segments):")
    print("-" * 95)
    for i in range(min(5, len(segments))):
        s = segments[i]
        print(f"\n[Seg {i}] {s['game_name']} | offset={s['offset']:.1f}s")
        print(f"  REF:  {normalizer(s['reference'])[:120]}")
        if s['entities']:
            ents = ", ".join(f"{t}:{n}" for t, n in s['entities'])
            print(f"  ENTS: {ents[:120]}")
        for strat in strategies:
            if strat in all_predictions:
                print(f"  {strat:>4}: {normalizer(all_predictions[strat][i])[:120]}")

    # Save JSON
    out_file = f"prompting_results_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": args.model, "model_path": model_path,
            "strategies": strategies, "n_segments": len(segments),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
