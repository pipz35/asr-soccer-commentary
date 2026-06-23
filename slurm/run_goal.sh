#!/bin/bash
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=goal_output_%j.log

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/philip/asr_project
source venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

python whisper_goal.py
