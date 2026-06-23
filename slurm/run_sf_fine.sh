#!/bin/bash
#SBATCH --job-name=whisper_sf2
#SBATCH --output=sf_fine_output_%j.log
#SBATCH --error=sf_fine_error_%j.log
#SBATCH --account=master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/philip/asr_project
source venv/bin/activate
export HF_HOME=/scratch/izar/philip/hf_cache

python whisper_shallow_fusion.py --model finetuned --strengths 0.5,1.0,1.5,2.0,3.0
