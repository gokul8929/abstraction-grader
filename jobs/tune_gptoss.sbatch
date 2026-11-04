#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH -c 16
#SBATCH -t 0-03:00:00
#SBATCH -J tunegpt
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-tune.out
# One tuning round for gpt-oss-120b (condition B): the 3 tuning queries x low/medium/high, seed 0 (9 runs).
# Usage:  sbatch --export=ALL,PROMPTS=gptoss_r1 jobs/tune_gptoss.sbatch
MODEL_KEY=gpt-oss-120b HF_ID=openai/gpt-oss-120b PORT=8001 SEEDS=0
PROMPTS=${PROMPTS:?set PROMPTS, e.g. sbatch --export=ALL,PROMPTS=gptoss_r1 jobs/tune_gptoss.sbatch}
TAG=tune-$PROMPTS QUERIES=stockindex:1,bookreview:1,yelp:1 SKIP_SMOKE=1 HF_EXCLUDE='--exclude original/* --exclude metal/*'
SERVE_ARGS="--tensor-parallel-size 4 --tool-call-parser openai"
source ~/abstraction-grader/jobs/_common.sh
