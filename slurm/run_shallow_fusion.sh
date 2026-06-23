#!/bin/bash
#SBATCH --job-name=whisper_sf
#SBATCH --output=sf_output_%j.log
#SBATCH --error=sf_error_%j.log
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

echo "=== Whisper Shallow Fusion ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python whisper_shallow_fusion.py --model finetuned --strengths 2.0,5.0,10.0,15.0

echo ""
echo "=== Done: $(date) ==="
