#!/bin/bash
#SBATCH --job-name=goal_save
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=goal_save_%j.log

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
source /scratch/izar/philip/asr_project/venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

cd /scratch/izar/philip/asr_project
python whisper_goal_save.py
