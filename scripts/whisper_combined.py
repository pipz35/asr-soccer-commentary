"""
Combined Prompting + Shallow Fusion Experiments
================================================
Tests: PL+greedy (existing), PL+beam=5, PL+beam=5+SF, SF only (beam=5)
"""

import os, sys, json, re, torch, librosa
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    SequenceBiasLogitsProcessor, LogitsProcessorList,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from jiwer import wer
from tqdm import tqdm

ORIGINAL_MODEL = "openai/whisper-medium"
FINETUNED_MODEL = "/scratch/izar/philip/asr_project/whisper_soccer_finetuned/final"
HF_CACHE = "/scratch/izar/philip/hf_cache"
AUDIO_DIR = "/scratch/izar/philip/soccernet_audio"
GT_DIR = "/scratch/izar/philip/asr_project/goal_data/commentaries"
TEST_MAX_OFFSET = 900

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
            "Aguero, Zabaleta, Iheanacho, Bony, Caballero, Shaqiri"
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
            "Willian, Fabregas, Moses, Batshuayi, Ivanovic, Oscar"
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
            "Muller, Robben, Lewandowski, Rafinha, Alaba, Gotze, Ribery, Thiago"
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
            "Matuidi, Motta, Rabiot, Pastore, Cavani, Lavezzi, Ibrahimovic, David Luiz"
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


def extract_entities(entity_text):
    if not entity_text:
        return []
    entities = []
    for match in re.finditer(r'\[(\w+)\](.*?)\[\1\]', entity_text):
        etype, ename = match.group(1), match.group(2).strip()
        if ename:
            entities.append((etype, ename))
    return entities


def load_test_segments():
    segments = []
    for game in TEST_GAMES:
        print(f"\nLoading: {game['name']}")
        game_loaded = 0
        game_skipped = 0
        for half in ["1", "2"]:
            gt_file = os.path.join(GT_DIR, game["gt_prefix"] + f"_{half}.json")
            if not os.path.exists(gt_file):
                continue
            with open(gt_file, "r") as f_gt:
                annotations = json.load(f_gt)
            audio_file = os.path.join(AUDIO_DIR, game["audio_path"], f"{half}_224p.wav")
            if not os.path.exists(audio_file):
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
                except:
                    game_skipped += 1
                    continue
                segments.append({
                    "audio": audio,
                    "reference": commentary.strip(),
                    "entity_text": entity_text,
                    "entities": extract_entities(entity_text),
                    "game_name": game["name"],
                    "roster": game["roster"],
                    "game_entities": game["entities"],
                    "offset": offset,
                    "duration": duration,
                })
                game_loaded += 1
        print(f"  Loaded: {game_loaded}, Skipped: {game_skipped}")
    print(f"\nTotal test segments: {len(segments)}")
    return segments


def build_bias_list(entity_names, tokenizer, strength):
    bias = []
    seen = set()
    for name in entity_names:
        tokens = tokenizer.encode(" " + name, add_special_tokens=False)
        key = tuple(tokens)
        if key not in seen:
            seen.add(key)
            bias.append([tokens, strength])
    return bias


def compute_metrics(segments, predictions, normalizer):
    refs = [normalizer(s["reference"]) for s in segments]
    preds = [normalizer(p) for p in predictions]
    valid = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    vr, vp = zip(*valid)
    global_wer = wer(list(vr), list(vp)) * 100

    # Entity WER
    erefs, epreds = [], []
    for s, p in zip(segments, predictions):
        if s["entities"]:
            r = normalizer(s["reference"])
            if r.strip():
                erefs.append(r)
                epreds.append(normalizer(p))
    entity_wer = wer(erefs, epreds) * 100 if erefs else 0

    # Entity detection
    full, partial = [], []
    for s, p in zip(segments, predictions):
        if not s["entities"]:
            continue
        pn = normalizer(p).lower()
        for et, en in s["entities"]:
            words = normalizer(en).lower().split()
            if words:
                full.append(1 if all(w in pn for w in words) else 0)
                partial.append(1 if any(w in pn for w in words) else 0)

    n = len(full)
    return {
        "global_wer": global_wer,
        "entity_wer": entity_wer,
        "detect_full": sum(full)/n*100 if n else 0,
        "detect_partial": sum(partial)/n*100 if n else 0,
        "n_entities": n,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = WhisperProcessor.from_pretrained(ORIGINAL_MODEL, cache_dir=HF_CACHE)
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
    model = WhisperForConditionalGeneration.from_pretrained(
        FINETUNED_MODEL, cache_dir=HF_CACHE
    ).to(device)
    model.eval()

    segments = load_test_segments()
    forced = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    # Pre-build bias lists per game
    game_biases = {}
    for game in TEST_GAMES:
        game_biases[game["name"]] = build_bias_list(
            game["entities"], processor.tokenizer, 3.0
        )

    # Define experiments
    experiments = [
        {"name": "N_greedy",           "prompt": False, "beams": 1, "bias": False},
        {"name": "N_beam5",            "prompt": False, "beams": 5, "bias": False},
        {"name": "PL_greedy",          "prompt": True,  "beams": 1, "bias": False},
        {"name": "PL_beam5",           "prompt": True,  "beams": 5, "bias": False},
        {"name": "SF3_beam5",          "prompt": False, "beams": 5, "bias": True},
        {"name": "PL+SF3_beam5",       "prompt": True,  "beams": 5, "bias": True},
    ]

    all_results = []
    all_preds = {}

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"  Prompt: {exp['prompt']}, Beams: {exp['beams']}, Bias: {exp['bias']}")
        print(f"{'='*60}")

        preds = []
        for seg in tqdm(segments, desc=f"  {exp['name']}"):
            inputs = processor(
                seg["audio"], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            gen_kwargs = {"forced_decoder_ids": forced}

            if exp["beams"] > 1:
                gen_kwargs["num_beams"] = exp["beams"]

            if exp["prompt"]:
                roster = seg["roster"]
                prompt_text = f"Football match commentary. Players: {roster}"
                prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
                gen_kwargs["prompt_ids"] = prompt_ids

            if exp["bias"]:
                bl = game_biases.get(seg["game_name"], [])
                if bl:
                    gen_kwargs["logits_processor"] = LogitsProcessorList([
                        SequenceBiasLogitsProcessor(sequence_bias=bl)
                    ])

            with torch.no_grad():
                ids = model.generate(inputs, **gen_kwargs)

            pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
            preds.append(pred)

        metrics = compute_metrics(segments, preds, normalizer)
        result = {"name": exp["name"], **metrics}
        all_results.append(result)
        all_preds[exp["name"]] = preds

        print(f"  Global WER:      {metrics['global_wer']:.2f}%")
        print(f"  Entity WER:      {metrics['entity_wer']:.2f}%")
        print(f"  Detect (full):   {metrics['detect_full']:.1f}%")
        print(f"  Detect (partial):{metrics['detect_partial']:.1f}%")

    # Summary
    print("\n\n" + "=" * 100)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<20} {'Global WER':>12} {'Entity WER':>12} {'Detect(full)':>14} {'Detect(part)':>14}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['name']:<20} {r['global_wer']:>11.2f}% {r['entity_wer']:>11.2f}% "
              f"{r['detect_full']:>13.1f}% {r['detect_partial']:>13.1f}%")
    print("=" * 100)

    # Sample predictions
    print("\n\nSAMPLE PREDICTIONS (first 5):")
    print("-" * 100)
    for i in range(min(5, len(segments))):
        s = segments[i]
        print(f"\n[Seg {i}] {s['game_name']} | offset={s['offset']:.1f}s")
        print(f"  REF:  {normalizer(s['reference'])[:120]}")
        if s['entities']:
            print(f"  ENTS: {', '.join(f'{t}:{n}' for t,n in s['entities'])[:120]}")
        for exp in experiments:
            if exp["name"] in all_preds:
                print(f"  {exp['name']:>18}: {normalizer(all_preds[exp['name']][i])[:120]}")

    with open("combined_results.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"\nSaved to combined_results.json")


if __name__ == "__main__":
    main()
