#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH -c 12
#SBATCH -t 1-12:00:00
#SBATCH -J B-glimmer
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-B-glimmer.out
# Condition B, Muse Glimmer-30B: base prompts (round 0 showed no protocol failures, so no tuning addenda).
# All 13 queries x low/medium/xhigh x seeds 0,1 = 78 runs (~9-10 h). Resumable: resubmitting skips finished runs.
MODEL_KEY=muse-glimmer-30b HF_ID=meta-models/Muse-Glimmer-30B PORT=8003 TAG=B-glimmer SEEDS=0,1 SKIP_SMOKE=1 HF_EXCLUDE=''
SERVE_ARGS="--tensor-parallel-size 2 --language-model-only --generation-config auto --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer"
source ~/abstraction-grader/jobs/_common.sh
