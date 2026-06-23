#!/bin/bash
#SBATCH --job-name=whisper_comb
#SBATCH --output=combined_output_%j.log
#SBATCH --error=combined_error_%j.log
#SBATCH --account=master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/philip/asr_project
source venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

echo "=== Combined Experiments ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python whisper_combined.py

echo "=== Done: $(date) ==="
