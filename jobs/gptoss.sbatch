#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -J gptoss
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-gptoss.out
# gpt-oss-120b (OpenAI): MXFP4 MoE, tensor parallel 4 on A100 40GB, settings low/medium/high, seeds 0 and 1 (78 runs).
MODEL_KEY=gpt-oss-120b HF_ID=openai/gpt-oss-120b PORT=8001 TAG=pilot2-gptoss SEEDS=0,1
HF_EXCLUDE='--exclude original/* metal/*'                 # skip the duplicate checkpoint copies
SERVE_ARGS="--tensor-parallel-size 4 --tool-call-parser openai"
source ~/abstraction-grader/jobs/_common.sh
