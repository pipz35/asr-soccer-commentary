#!/bin/bash
#SBATCH --job-name=whisper_finetune
#SBATCH --output=finetune_output_%j.log
#SBATCH --error=finetune_error_%j.log
#SBATCH --account=master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/philip/asr_project
source venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

python finetune_whisper_soccer.py
