#!/bin/bash
#SBATCH --job-name=whisper_eval
#SBATCH --output=eval_output_%j.log
#SBATCH --error=eval_error_%j.log
#SBATCH --account=master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/philip/asr_project
source venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

python evaluate_finetuned.py
