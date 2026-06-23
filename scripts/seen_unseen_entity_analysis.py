"""
Seen vs Unseen Entity Analysis for Soccer Commentary ASR
=========================================================
Runs fine-tuned Whisper with N, PL, and PL+SF3 configurations,
then breaks down entity detection accuracy by:
  - Seen entities (appeared in the 16 training games)
  - Unseen entities (only in test games, never seen during fine-tuning)

Also breaks down by entity token length (1-token vs 2+ tokens).

Uses train_entity_names.json (generated separately) for the seen/unseen split.
"""

import os
import sys
import json
import re
import torch
import librosa
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    SequenceBiasLogitsProcessor, LogitsProcessorList,
)
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
TRAIN_ENTITIES_FILE = "/scratch/izar/philip/asr_project/train_entity_names.json"

TEST_MAX_OFFSET = 900

# Rosters and entity lists matching combined/SF scripts
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
        "entities": [
            "Mignolet", "Clyne", "Flanagan", "Lovren", "Can", "Henderson",
            "Lallana", "Coutinho", "Firmino", "Milner", "Sturridge", "Origi",
            "Ibe", "Benteke", "Allen", "Skrtel", "Kolo Toure", "Yaya Toure",
            "Hart", "Sagna", "Otamendi", "Kompany", "Kolarov", "Clichy",
            "Fernandinho", "Fernando", "Silva", "David Silva", "Navas",
            "Sterling", "Aguero", "Zabaleta", "Iheanacho", "Bony", "Caballero",
            "Shaqiri", "Liverpool", "Manchester City", "Leicester",
            "Klopp", "Pellegrini",
        ],
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
        "entities": [
            "Cech", "Bellerin", "Mustafi", "Koscielny", "Monreal", "Coquelin",
            "Cazorla", "Walcott", "Ozil", "Iwobi", "Sanchez", "Xhaka",
            "Elneny", "Giroud", "Gibbs", "Oxlade Chamberlain", "Lucas Perez",
            "Courtois", "Azpilicueta", "David Luiz", "Cahill", "Alonso",
            "Kante", "Matic", "Pedro", "Hazard", "Costa", "Diego Costa",
            "Willian", "Fabregas", "Moses", "Batshuayi", "Ivanovic", "Oscar",
            "Arsenal", "Chelsea", "Conte", "Wenger",
        ],
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
        "entities": [
            "Ter Stegen", "Dani Alves", "Pique", "Mascherano", "Jordi Alba",
            "Busquets", "Rakitic", "Iniesta", "Messi", "Suarez", "Neymar",
            "Xavi", "Bartra", "Neuer", "Boateng", "Benatia", "Bernat",
            "Xabi Alonso", "Lahm", "Schweinsteiger", "Muller", "Robben",
            "Lewandowski", "Rafinha", "Alaba", "Gotze", "Ribery", "Thiago",
            "Barcelona", "Bayern Munich", "Bayern", "Guardiola", "Luis Enrique",
        ],
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
        "entities": [
            "Mandanda", "Fanni", "Morel", "Mendy", "Romao", "Imbula",
            "Lemina", "Payet", "Alessandrini", "Ayew", "Gignac", "Batshuayi",
            "Thauvin", "Ocampos", "Djedje", "Sirigu", "Van der Wiel",
            "Marquinhos", "Thiago Silva", "Maxwell", "Verratti", "Matuidi",
            "Motta", "Rabiot", "Pastore", "Cavani", "Lavezzi", "Ibrahimovic",
            "David Luiz", "Marseille", "PSG", "Paris Saint Germain",
            "Bielsa", "Laurent Blanc",
        ],
    },
]

BIAS_STRENGTH = 3.0


# ============================================================
# Helpers
# ============================================================

def extract_entities(entity_text):
    if not entity_text:
        return []
    entities = []
    for match in re.finditer(r'\[(\w+)\](.*?)\[\1\]', entity_text):
        etype, ename = match.group(1), match.group(2).strip()
        if ename:
            entities.append((etype, ename))
    return entities


def is_clean_entity(name):
    name_lower = name.lower()
    if len(name_lower) < 2 or len(name_lower) > 40:
        return False
    if '[' in name_lower or ']' in name_lower:
        return False
    if '.' in name_lower or ',' in name_lower:
        return False
    if not name_lower[0].isalpha():
        return False
    return True


def load_train_entity_set():
    with open(TRAIN_ENTITIES_FILE, "r") as f:
        data = json.load(f)
    return set(data["train_entities"])


def load_test_segments():
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
                if offset >= TEST_MAX_OFFSET:
                    continue
                if duration > 30:
                    game_skipped += 1
                    continue
                if not commentary.strip():
                    game_skipped += 1
                    continue
                try:
                    audio, sr = librosa.load(audio_file, sr=16000,
                                             offset=offset, duration=duration)
                    if len(audio) < 160:
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


def build_sequence_bias(entity_names, tokenizer, bias_strength):
    bias = []
    seen = set()
    for name in entity_names:
        tokens = tokenizer.encode(" " + name, add_special_tokens=False)
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        bias.append([tokens, bias_strength])
    return bias


# ============================================================
# Entity Detection with Seen/Unseen Split
# ============================================================

def compute_entity_detection_split(segments, predictions, normalizer,
                                   train_entity_set):
    """Compute entity detection accuracy split by seen/unseen and by token length."""
    results = {
        "all":    {"full": [], "partial": [], "details": []},
        "seen":   {"full": [], "partial": [], "details": []},
        "unseen": {"full": [], "partial": [], "details": []},
        # By token count (number of words in entity name)
        "1_token":  {"full": [], "partial": [], "details": []},
        "2plus_token": {"full": [], "partial": [], "details": []},
    }

    for seg, pred in zip(segments, predictions):
        if not seg["entities"]:
            continue

        pred_norm = normalizer(pred).lower()

        for etype, ename in seg["entities"]:
            if not is_clean_entity(ename):
                continue

            ename_norm = normalizer(ename).lower()
            words = ename_norm.split()
            if not words:
                continue

            all_found = all(w in pred_norm for w in words)
            any_found = any(w in pred_norm for w in words)

            is_seen = ename_norm in train_entity_set

            detail = {
                "type": etype, "name": ename, "name_norm": ename_norm,
                "full": all_found, "partial": any_found,
                "seen_in_training": is_seen,
                "n_words": len(words),
                "pred_snippet": pred[:100],
            }

            # All
            results["all"]["full"].append(1 if all_found else 0)
            results["all"]["partial"].append(1 if any_found else 0)
            results["all"]["details"].append(detail)

            # Seen vs unseen
            bucket = "seen" if is_seen else "unseen"
            results[bucket]["full"].append(1 if all_found else 0)
            results[bucket]["partial"].append(1 if any_found else 0)
            results[bucket]["details"].append(detail)

            # Token length
            tok_bucket = "1_token" if len(words) == 1 else "2plus_token"
            results[tok_bucket]["full"].append(1 if all_found else 0)
            results[tok_bucket]["partial"].append(1 if any_found else 0)
            results[tok_bucket]["details"].append(detail)

    # Compute percentages
    summary = {}
    for key in results:
        n = len(results[key]["full"])
        if n > 0:
            summary[key] = {
                "n": n,
                "full_pct": sum(results[key]["full"]) / n * 100,
                "partial_pct": sum(results[key]["partial"]) / n * 100,
            }
        else:
            summary[key] = {"n": 0, "full_pct": 0, "partial_pct": 0}

    return summary, results


# ============================================================
# Run Configurations
# ============================================================

def run_config(model, processor, normalizer, segments, config_name,
               use_prompt=False, use_bias=False, game_biases=None):
    """Run one configuration across all segments."""
    device = next(model.parameters()).device
    predictions = []

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    for seg in tqdm(segments, desc=f"  {config_name}"):
        inputs = processor(
            seg["audio"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        generate_kwargs = {"forced_decoder_ids": forced_decoder_ids}

        # Prompting
        if use_prompt and seg.get("roster"):
            prompt_text = f"Football match commentary. Players: {seg['roster']}"
            prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt")
            prompt_ids = prompt_ids.to(device)
            generate_kwargs["prompt_ids"] = prompt_ids

        # Shallow fusion
        if use_bias and game_biases:
            bias_list = game_biases.get(seg["game_name"], [])
            if bias_list:
                logits_processor = LogitsProcessorList([
                    SequenceBiasLogitsProcessor(sequence_bias=bias_list)
                ])
                generate_kwargs["logits_processor"] = logits_processor
                generate_kwargs["num_beams"] = 5

        # If using prompt + beam search (no bias), still use beam search for fair comparison
        if use_prompt and not use_bias:
            # Greedy for prompt-only (matching prompting experiment setup)
            pass

        with torch.no_grad():
            predicted_ids = model.generate(inputs, **generate_kwargs)

        pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(pred)

    # Global WER
    refs_norm = [normalizer(seg["reference"]) for seg in segments]
    preds_norm = [normalizer(p) for p in predictions]
    valid = [(r, p) for r, p in zip(refs_norm, preds_norm) if r.strip()]
    if valid:
        valid_refs, valid_preds = zip(*valid)
        global_wer_val = wer(list(valid_refs), list(valid_preds)) * 100
    else:
        global_wer_val = 0.0

    return global_wer_val, predictions


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SEEN vs UNSEEN ENTITY ANALYSIS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load train entity set
    print("\nLoading training entity set...")
    train_entity_set = load_train_entity_set()
    print(f"  Training entities: {len(train_entity_set)}")

    # Load processor + normalizer
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(ORIGINAL_MODEL, cache_dir=HF_CACHE)
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
    tokenizer = processor.tokenizer

    # Load model
    print("Loading fine-tuned model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        FINETUNED_MODEL, cache_dir=HF_CACHE
    ).to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test segments
    print("\n--- Loading test segments ---")
    segments = load_test_segments()
    if not segments:
        print("ERROR: No segments loaded!")
        sys.exit(1)

    # Build biases for SF
    print("\nBuilding sequence biases for shallow fusion...")
    game_biases = {}
    for game in TEST_GAMES:
        bias = build_sequence_bias(game["entities"], tokenizer, BIAS_STRENGTH)
        game_biases[game["name"]] = bias
        print(f"  {game['name']}: {len(bias)} biased sequences")

    # Define configurations to test
    configs = [
        ("N_greedy",    False, False),   # No prompt, no bias, greedy
        ("PL_greedy",   True,  False),   # Player list prompt, greedy
        ("PL+SF3_beam5", True, True),    # Player list + shallow fusion bias=3.0, beam=5
    ]

    all_results = {}

    for config_name, use_prompt, use_bias in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")

        global_wer_val, predictions = run_config(
            model, processor, normalizer, segments, config_name,
            use_prompt=use_prompt, use_bias=use_bias,
            game_biases=game_biases if use_bias else None,
        )

        summary, details = compute_entity_detection_split(
            segments, predictions, normalizer, train_entity_set
        )

        all_results[config_name] = {
            "global_wer": global_wer_val,
            "entity_summary": summary,
        }

        print(f"\n  Global WER: {global_wer_val:.2f}%")
        print(f"\n  {'Category':<16} {'N':>6} {'Full%':>8} {'Partial%':>10}")
        print(f"  {'-'*42}")
        for key in ["all", "seen", "unseen", "1_token", "2plus_token"]:
            s = summary[key]
            print(f"  {key:<16} {s['n']:>6} {s['full_pct']:>7.1f}% {s['partial_pct']:>9.1f}%")

    # Final comparison table
    print("\n\n" + "=" * 90)
    print("FINAL COMPARISON: SEEN vs UNSEEN ENTITY DETECTION (Full Match %)")
    print("=" * 90)
    print(f"{'Config':<18} {'WER':>7} {'All':>8} {'Seen':>8} {'Unseen':>8} {'Gap':>8} {'1-tok':>8} {'2+-tok':>8}")
    print("-" * 90)
    for config_name in all_results:
        r = all_results[config_name]
        s = r["entity_summary"]
        gap = s["seen"]["full_pct"] - s["unseen"]["full_pct"]
        print(f"{config_name:<18} {r['global_wer']:>6.2f}% "
              f"{s['all']['full_pct']:>7.1f}% "
              f"{s['seen']['full_pct']:>7.1f}% "
              f"{s['unseen']['full_pct']:>7.1f}% "
              f"{gap:>+7.1f}% "
              f"{s['1_token']['full_pct']:>7.1f}% "
              f"{s['2plus_token']['full_pct']:>7.1f}%")
    print("=" * 90)
    print(f"\nEntity counts — Seen: {all_results[list(all_results.keys())[0]]['entity_summary']['seen']['n']}, "
          f"Unseen: {all_results[list(all_results.keys())[0]]['entity_summary']['unseen']['n']}")

    # Save detailed results
    out_file = "seen_unseen_analysis_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
