import json
import os
import soundfile as sf
import numpy as np
from datasets import Dataset, Audio, DatasetDict

GOAL_DIR = "/scratch/izar/philip/asr_project/goal_data/commentaries"
AUDIO_DIR = "/scratch/izar/philip/soccernet_audio"
OUTPUT_DIR = "/scratch/izar/philip/asr_project/soccer_finetune_dataset"
SAMPLE_RATE = 16000

# Map every GOAL JSON to its SoccerNet audio WAV
GAME_MAPPING = {
    # EPL games
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
    # Champions League games
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
    # Other leagues
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_1.json": "france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/1_224p.wav",
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_2.json": "france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/2_224p.wav",
    "italy_serie-a__2014-2015__2015-02-15_-_14-30_AC_Milan_1_-_1_Empoli_1.json": "italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/1_224p.wav",
    "italy_serie-a__2014-2015__2015-02-15_-_14-30_AC_Milan_1_-_1_Empoli_2.json": "italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/2_224p.wav",
    "spain_laliga__2014-2015__2015-05-02_-_21-00_Sevilla_2_-_3_Real_Madrid_1.json": "spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid/1_224p.wav",
    "spain_laliga__2014-2015__2015-05-02_-_21-00_Sevilla_2_-_3_Real_Madrid_2.json": "spain_laliga/2014-2015/2015-05-02 - 21-00 Sevilla 2 - 3 Real Madrid/2_224p.wav",
}

# TEST GAMES: 4 diverse games (first 15 min = 900 seconds only)
# Mix of EPL + Champions League + different teams for coverage
TEST_GAMES = [
    # Liverpool vs Man City (your original baseline game)
    "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City_1.json",
    "england_epl__2015-2016__2016-03-02_-_23-00_Liverpool_3_-_0_Manchester_City_2.json",
    # Arsenal vs Chelsea (different teams/speakers)
    "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea_1.json",
    "england_epl__2016-2017__2016-09-24_-_19-30_Arsenal_3_-_0_Chelsea_2.json",
    # Barcelona vs Bayern (Champions League, different vibe)
    "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich_1.json",
    "europe_uefa-champions-league__2014-2015__2015-05-06_-_21-45_Barcelona_3_-_0_Bayern_Munich_2.json",
    # Marseille vs PSG (Ligue 1, different league)
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_1.json",
    "france_ligue-1__2014-2015__2015-04-05_-_22-00_Marseille_2_-_3_Paris_SG_2.json",
]

TEST_MAX_OFFSET = 900  # First 15 minutes


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
            print(f"  Skipping segment at {offset:.1f}s: duration {duration:.1f}s > 30s")
            continue
        segments.append({"offset": offset, "duration": duration, "commentary": commentary})
    return segments


def extract_audio(wav_path, offset, duration):
    start = int(offset * SAMPLE_RATE)
    stop = int((offset + duration) * SAMPLE_RATE)
    audio, sr = sf.read(wav_path, start=start, stop=stop, dtype='float32')
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}, got {sr}"
    return audio


def main():
    train_samples = []
    test_samples = []

    for json_name, wav_rel in GAME_MAPPING.items():
        json_path = os.path.join(GOAL_DIR, json_name)
        wav_path = os.path.join(AUDIO_DIR, wav_rel)

        if not os.path.exists(json_path):
            print(f"MISSING JSON: {json_path}")
            continue
        if not os.path.exists(wav_path):
            print(f"MISSING WAV: {wav_path}")
            continue

        is_test = json_name in TEST_GAMES
        max_off = TEST_MAX_OFFSET if is_test else None
        label = "TEST" if is_test else "TRAIN"

        print(f"[{label}] {json_name}")
        segments = load_segments(json_path, max_offset=max_off)
        print(f"  {len(segments)} segments")

        for seg in segments:
            try:
                audio = extract_audio(wav_path, seg["offset"], seg["duration"])
                sample = {
                    "audio": {"array": audio, "sampling_rate": SAMPLE_RATE},
                    "sentence": seg["commentary"],
                }
                if is_test:
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)
            except Exception as e:
                print(f"  Error at {seg['offset']:.1f}s: {e}")

    train_hrs = sum(len(s["audio"]["array"]) / SAMPLE_RATE for s in train_samples) / 3600
    test_hrs = sum(len(s["audio"]["array"]) / SAMPLE_RATE for s in test_samples) / 3600

    print(f"\n{'='*50}")
    print(f"Train: {len(train_samples)} samples ({train_hrs:.2f} hours)")
    print(f"Test:  {len(test_samples)} samples ({test_hrs:.2f} hours)")
    print(f"{'='*50}")

    train_ds = Dataset.from_list(train_samples).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    test_ds = Dataset.from_list(test_samples).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    ds = DatasetDict({"train": train_ds, "test": test_ds})
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
