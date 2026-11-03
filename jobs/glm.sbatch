#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -J glmair
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-glm.out
# GLM-4.5-Air (Zhipu AI): FP8 MoE, tensor parallel 4 on A100 40GB, thinking on/off, seeds 0 and 1 (52 runs).
MODEL_KEY=glm-4.5-air HF_ID=zai-org/GLM-4.5-Air-FP8 PORT=8002 TAG=pilot2-glm-v3 SEEDS=0,1
HF_EXCLUDE=''
SERVE_ARGS="--tensor-parallel-size 4 --tool-call-parser glm45 --reasoning-parser glm45"
source ~/abstraction-grader/jobs/_common.sh
