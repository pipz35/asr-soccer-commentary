# Improving Domain-Specific ASR for Soccer Broadcast Commentary

Fine-tuning, prompting, and shallow fusion for entity recognition in soccer broadcast commentary using OpenAI's Whisper.

**Author:** Philip Awada
**Supervisor:** Dr. Petr Motlicek (EPFL / Idiap Research Institute)
**Project type:** Semester project, EPFL, Spring 2026

## Overview

Pre-trained ASR models like Whisper degrade significantly on soccer commentary (17.09% WER vs. 2.43% on clean speech). This project explores three techniques to close the gap, with a focus on entity recognition (player names, team names):

1. **Fine-tuning** Whisper Medium on 15.66 hours of soccer commentary → 13.53% WER (21% relative improvement)
2. **Decoder prompting** with match rosters → 84.8% entity detection accuracy
3. **Shallow fusion** via sequence biasing → complementary gains
4. **Combined approach** → 88.3% entity detection accuracy (within 2pp of oracle)

Key finding: fine-tuning is a prerequisite for inference-time techniques in this domain. Without it, prompting causes severe hallucination (WER > 282%).

## Repository Structure
## Datasets

- **LibriSpeech** — public, via HuggingFace (`openslr/librispeech_asr`)
- **SoccerNet** — requires NDA ([soccer-net.org](https://www.soccer-net.org/))
- **GOAL benchmark** — public, BSD-3-Clause ([GitHub](https://github.com/THU-KEG/goal))

## Quick Start (EPFL Izar)

See the companion **Reproducibility Guide** for full instructions. In short:

```bash
ssh <username>@izar.hpc.epfl.ch
module load gcc/11.3.0 python/3.10.4 cuda/12.1.1
cd /scratch/izar/<username>/asr_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sbatch slurm/run_finetune.sh
```

## Key Results

| Configuration | WER | Entity Det. Accuracy |
|---|---|---|
| Whisper Medium (zero-shot) | 17.09% | 63.9% |
| + Fine-tuning | 13.53% | 75.8% |
| + Prompting (roster) | 12.17% | 84.8% |
| + Prompting + Shallow Fusion | 12.49% | 88.3% |
| Oracle (ground truth prompt) | 12.17% | 90.3% |

## References

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML 2023
- Piaget, "Improving ASR and Callsign Detection in ATC Speech using Whisper Prompting," Idiap-RR-04-2025
- Qi et al., "GOAL: A Challenging Knowledge-grounded Video Captioning Benchmark," CIKM 2023
