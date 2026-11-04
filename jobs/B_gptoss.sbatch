#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -J B-gptoss
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-B-gptoss.out
# Condition B (tuned system), gpt-oss-120b: prompts gptoss_r3 + tool_choice=required on orchestrator turns.
# All 13 queries x low/medium/high x seeds 0,1 = 78 runs. Resumable: resubmitting skips finished runs.
MODEL_KEY=gpt-oss-120b HF_ID=openai/gpt-oss-120b PORT=8001 TAG=B-gptoss SEEDS=0,1 PROMPTS=gptoss_r3
HF_EXCLUDE='--exclude original/* --exclude metal/*'
SERVE_ARGS="--tensor-parallel-size 4 --tool-call-parser openai"
source ~/abstraction-grader/jobs/_common.sh
