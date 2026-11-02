#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH -J qwens1
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-qwen.out
# Qwen3.8-27B second seed: same settings as pilot1, tag pilot1, seed 1 only (39 runs; seed 0 is already done).
MODEL_KEY=qwen3.8-27b HF_ID=Qwen/Qwen3.8-27B PORT=8000 TAG=pilot1 SEEDS=1
HF_EXCLUDE=''
SERVE_ARGS="--tensor-parallel-size 2 --language-model-only --reasoning-parser qwen3 --tool-call-parser qwen3_xml"
source ~/abstraction-grader/jobs/_common.sh
