#!/bin/bash
# Extract audio from all SoccerNet .mkv files to 16kHz mono WAV

FFMPEG=/scratch/izar/philip/asr_project/ffmpeg
DATA_DIR=/scratch/izar/philip/soccernet_data
OUTPUT_DIR=/scratch/izar/philip/soccernet_audio

mkdir -p "$OUTPUT_DIR"

find "$DATA_DIR" -name "*.mkv" -print0 | while IFS= read -r -d '' mkv_file; do
    rel_path="${mkv_file#$DATA_DIR/}"
    wav_path="$OUTPUT_DIR/${rel_path%.mkv}.wav"
    wav_dir=$(dirname "$wav_path")
    mkdir -p "$wav_dir"

    if [ -f "$wav_path" ]; then
        echo "SKIP (exists): $wav_path"
        continue
    fi

    echo "Extracting: $mkv_file"
    "$FFMPEG" -nostdin -i "$mkv_file" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$wav_path" -y -loglevel error < /dev/null
done

echo "Done!"
du -sh "$OUTPUT_DIR"
