#!/bin/bash
#SBATCH --job-name=whisper_prompt
#SBATCH --output=prompt_output_%j.log
#SBATCH --error=prompt_error_%j.log
#SBATCH --account=master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/philip/asr_project
source venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

echo "=== Whisper Prompting Experiments ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python whisper_prompting_experiments.py --model finetuned --strategies N,C,PL,GT,RT

echo ""
echo "=== Done: $(date) ==="
