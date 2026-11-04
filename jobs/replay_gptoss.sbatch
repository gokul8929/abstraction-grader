#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH -c 16
#SBATCH -t 0-01:00:00
#SBATCH -J replaygpt
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-replay.out
# Replays the empty orchestrator turns of a tuning round and prints the raw tokens gpt-oss generated.
# Usage:  sbatch jobs/replay_gptoss.sbatch     (or --export=ALL,REPLAY_TAGS="tune-gptoss_r3")
MODEL_KEY=gpt-oss-120b HF_ID=openai/gpt-oss-120b PORT=8001 HF_EXCLUDE='--exclude original/* --exclude metal/*'
TAG=replay SERVE_ARGS="--tensor-parallel-size 4 --tool-call-parser openai"
RUN_CMD='for t in ${REPLAY_TAGS:-tune-gptoss_r2 tune-gptoss_r3}; do python -m grader.replay_raw --model gpt-oss-120b --tag $t; done'
source ~/abstraction-grader/jobs/_common.sh
