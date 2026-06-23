
import json
import os
import soundfile as sf
import numpy as np
from datasets import Dataset, Audio, DatasetDict, concatenate_datasets

GOAL_DIR = "/scratch/izar/philip/asr_project/goal_data/commentaries"
AUDIO_DIR = "/scratch/izar/philip/soccernet_audio"
OUTPUT_DIR = "/scratch/izar/philip/asr_project/soccer_finetune_dataset"
SAMPLE_RATE = 16000

GAME_MAPPING = {
    "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City_1.json": "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City/1_224p.wav",
    "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City_2.json": "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City/2_224p.wav",
    "england_epl__2015-2016__2016-03-20_-_19-00_Manchester_City_0_-_1_Manchester_United_1.json": "england_epl/2015-2016/2016-03-20 - 19-00 Manchester City 0 - 1 Manchester United/1_224p.wav",
    "england_epl__2015-2016__2016-03-20_-_19-00_Manchester_City_0_-_1_Manchester_United_2.json": "england_epl/2015-2016/2016-03-20 - 19-00 Manchester City 0 - 1 Manchester United/2_224p.wav",
    "england_epl__2016-2017__2016-08-14_-_18-00_Arsenal_3_-_4_Liverpool_1.json": "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.wav",
    "england_epl__2016-2017__2016-08-14_-_18-00_Arsenal_3_-_4_Liverpool_2.json": "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/2_224p.wav",
    "england_epl__2016-2017__2016-08-27_-_14-30_Tottenham_1_-_1_Liverpool_1.json": "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool/1_224p.wav",
    "england_epl__2016-2017__2016-08-27_-_14-30_Tottenham_1_-_1_Liverpool_2.json": "england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool/2_224p.wav",
    "england_epl__2016-2017__2016-09-16_-_22-00_Chelsea_1_-_2_Liverpool_1.json": "england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/1_224p.wav",
    "england_epl__2016-2017__2016-09-16_-_22-00_Chelsea_1_-_2_Liverpool_2.json": "england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/2_224p.wav",
    "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea_1.json": "england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea/1_224p.wav",
    "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea_2.json": "england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea/2_224p.wav",
    "england_epl__2016-2017__2016-10-17_-_22-00_Liverpool_0_-_0_Manchester_United_1.json": "england_epl/2016-2017/2016-10-17 - 22-00 Liverpool 0 - 0 Manchester United/1_224p.wav",
    "england_epl__2016-2017__2016-10-17_-_22-00_Liverpool_0_-_0_Manchester_United_2.json": "england_epl/2016-2017/2016-10-17 - 22-00 Liverpool 0 - 0 Manchester United/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2014-11-04_-_22-45_Real_Madrid_1_-_0_Liverpool_1.json": "europe_uefa-champions-league/2014-2015/2014-11-04 - 22-45 Real Madrid 1 - 0 Liverpool/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2014-11-04_-_22-45_Real_Madrid_1_-_0_Liverpool_2.json": "europe_uefa-champions-league/2014-2015/2014-11-04 - 22-45 Real Madrid 1 - 0 Liverpool/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2014-12-10_-_22-45_Barcelona_3_-_1_Paris_SG_1.json": "europe_uefa-champions-league/2014-2015/2014-12-10 - 22-45 Barcelona 3 - 1 Paris SG/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2014-12-10_-_22-45_Barcelona_3_-_1_Paris_SG_2.json": "europe_uefa-champions-league/2014-2015/2014-12-10 - 22-45 Barcelona 3 - 1 Paris SG/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-02-17_-_22-45_Paris_SG_1_-_1_Chelsea_1.json": "europe_uefa-champions-league/2014-2015/2015-02-17 - 22-45 Paris SG 1 - 1 Chelsea/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-02-17_-_22-45_Paris_SG_1_-_1_Chelsea_2.json": "europe_uefa-champions-league/2014-2015/2015-02-17 - 22-45 Paris SG 1 - 1 Chelsea/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-02-24_-_22-45_Manchester_City_1_-_2_Barcelona_1.json": "europe_uefa-champions-league/2014-2015/2015-02-24 - 22-45 Manchester City 1 - 2 Barcelona/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-02-24_-_22-45_Manchester_City_1_-_2_Barcelona_2.json": "europe_uefa-champions-league/2014-2015/2015-02-24 - 22-45 Manchester City 1 - 2 Barcelona/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-04-15_-_21-45_Paris_SG_1_-_3_Barcelona_1.json": "europe_uefa-champions-league/2014-2015/2015-04-15 - 21-45 Paris SG 1 - 3 Barcelona/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-04-15_-_21-45_Paris_SG_1_-_3_Barcelona_2.json": "europe_uefa-champions-league/2014-2015/2015-04-15 - 21-45 Paris SG 1 - 3 Barcelona/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-04-22_-_21-45_Real_Madrid_1_-_0_Atl._Madrid_1.json": "europe_uefa-champions-league/2014-2015/2015-04-22 - 21-45 Real Madrid 1 - 0 Atl. Madrid/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-04-22_-_21-45_Real_Madrid_1_-_0_Atl._Madrid_2.json": "europe_uefa-champions-league/2014-2015/2015-04-22 - 21-45 Real Madrid 1 - 0 Atl. Madrid/2_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich_1.json": "europe_uefa-champions-league/2014-2015/2015-05-06 - 21-45 Barcelona 3 - 0 Bayern Munich/1_224p.wav",
    "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich_2.json": "europe_uefa-champions-league/2014-2015/2015-05-06 - 21-45 Barcelona 3 - 0 Bayern Munich/2_224p.wav",
    "europe_uefa-champions-league__2015-2016__2015-09-15_-_21-45_Manchester_City_1_-_2_Juventus_1.json": "europe_uefa-champions-league/2015-2016/2015-09-15 - 21-45 Manchester City 1 - 2 Juventus/1_224p.wav",
    "europe_uefa-champions-league__2015-2016__2015-09-15_-_21-45_Manchester_City_1_-_2_Juventus_2.json": "europe_uefa-champions-league/2015-2016/2015-09-15 - 21-45 Manchester City 1 - 2 Juventus/2_224p.wav",
    "europe_uefa-champions-league__2015-2016__2015-11-04_-_22-45_Bayern_Munich_5_-_1_Arsenal_1.json": "europe_uefa-champions-league/2015-2016/2015-11-04 - 22-45 Bayern Munich 5 - 1 Arsenal/1_224p.wav",
    "europe_uefa-champions-league__2015-2016__2015-11-04_-_22-45_Bayern_Munich_5_-_1_Arsenal_2.json": "europe_uefa-champions-league/2015-2016/2015-11-04 - 22-45 Bayern Munich 5 - 1 Arsenal/2_224p.wav",
    "europe_uefa-champions-league__2015-2016__2015-11-25_-_22-45_Juventus_1_-_0_Manchester_City_1.json": "europe_uefa-champions-league/2015-2016/2015-11-25 - 22-45 Juventus 1 - 0 Manchester City/1_224p.wav",
    "europe_uefa-champions-league__2015-2016__2015-11-25_-_22-45_Juventus_1_-_0_Manchester_City_2.json": "europe_uefa-champions-league/2015-2016/2015-11-25 - 22-45 Juventus 1 - 0 Manchester City/2_224p.wav",
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_1.json": "france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/1_224p.wav",
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_2.json": "france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/2_224p.wav",
    "italy_serie-a__2014-2015__2015-02-15_-_14-30_AC_Milan_1_-_1_Empoli_1.json": "italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/1_224p.wav",
    "italy_serie-a__2014-2015__2015-02-15_-_14-30_AC_Milan_1_-_1_Empoli_2.json": "italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/2_224p.wav",
    "spain_laliga__2014-2015__2015-05-02_-_21-00_Sevilla_2_-_3_Real_Madrid_1.json": "spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid/1_224p.wav",
    "spain_laliga__2014-2015__2015-05-02_-_21-00_Sevilla_2_-_3_Real_Madrid_2.json": "spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid/2_224p.wav",
}

TEST_GAMES = [
    "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City_1.json",
    "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City_2.json",
    "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea_1.json",
    "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea_2.json",
    "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich_1.json",
    "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich_2.json",
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_1.json",
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_2.json",
]

TEST_MAX_OFFSET = 900


def load_segments(json_path, max_offset=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    segments = []
    for item in data:
        offset = item.get("offset", 0)
        duration = item.get("duration", 0)
        commentary = item.get("commentary", "").strip()
        if not commentary:
            continue
        if max_offset is not None and offset > max_offset:
            continue
        if duration > 30:
            continue
        segments.append({"offset": offset, "duration": duration, "commentary": commentary})
    return segments


def save_samples_for_game(json_name, wav_rel, is_test, output_base):
    """Save audio segments as individual WAV files + a metadata JSON per game."""
    json_path = os.path.join(GOAL_DIR, json_name)
    wav_path = os.path.join(AUDIO_DIR, wav_rel)

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        print(f"  MISSING: {json_path} or {wav_path}")
        return 0, 0.0

    max_off = TEST_MAX_OFFSET if is_test else None
    segments = load_segments(json_path, max_offset=max_off)

    split = "test" if is_test else "train"
    split_dir = os.path.join(output_base, split)
    os.makedirs(split_dir, exist_ok=True)

    game_id = json_name.replace(".json", "")
    metadata = []
    total_dur = 0.0

    for i, seg in enumerate(segments):
        try:
            start = int(seg["offset"] * SAMPLE_RATE)
            stop = int((seg["offset"] + seg["duration"]) * SAMPLE_RATE)
            audio, sr = sf.read(wav_path, start=start, stop=stop, dtype='float32')

            # Save individual WAV
            wav_name = f"{game_id}_seg{i:04d}.wav"
            wav_out = os.path.join(split_dir, wav_name)
            sf.write(wav_out, audio, SAMPLE_RATE)

            metadata.append({
                "file": wav_name,
                "sentence": seg["commentary"],
                "duration": seg["duration"]
            })
            total_dur += seg["duration"]
        except Exception as e:
            print(f"  Error at {seg['offset']:.1f}s: {e}")

    # Save metadata for this game
    meta_path = os.path.join(split_dir, f"{game_id}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    return len(metadata), total_dur


def build_hf_dataset(output_base):
    """Build HF dataset from saved WAV files + metadata JSONs."""
    for split in ["train", "test"]:
        split_dir = os.path.join(output_base, split)
        meta_files = sorted([f for f in os.listdir(split_dir) if f.endswith("_meta.json")])

        all_files = []
        all_sentences = []

        for mf in meta_files:
            with open(os.path.join(split_dir, mf)) as f:
                meta = json.load(f)
            for item in meta:
                all_files.append(os.path.join(split_dir, item["file"]))
                all_sentences.append(item["sentence"])

        ds = Dataset.from_dict({
            "audio": all_files,
            "sentence": all_sentences
        }).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

        ds.save_to_disk(os.path.join(output_base, f"{split}_dataset"))
        print(f"  {split}: {len(ds)} samples saved")


def main():
    output_base = OUTPUT_DIR
    os.makedirs(output_base, exist_ok=True)

    total_train = 0
    total_test = 0
    train_hrs = 0.0
    test_hrs = 0.0

    for json_name, wav_rel in GAME_MAPPING.items():
        is_test = json_name in TEST_GAMES
        label = "TEST" if is_test else "TRAIN"
        print(f"[{label}] {json_name}")

        n, dur = save_samples_for_game(json_name, wav_rel, is_test, output_base)
        print(f"  {n} segments ({dur/60:.1f} min)")

        if is_test:
            total_test += n
            test_hrs += dur / 3600
        else:
            total_train += n
            train_hrs += dur / 3600

    print(f"\n{'='*50}")
    print(f"Train: {total_train} samples ({train_hrs:.2f} hours)")
    print(f"Test:  {total_test} samples ({test_hrs:.2f} hours)")
    print(f"{'='*50}")

    print("\nBuilding HF datasets from saved files...")
    build_hf_dataset(output_base)
    print("Done!")


if __name__ == "__main__":
    main()
