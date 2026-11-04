#!/bin/bash
#SBATCH -p u22
#SBATCH -w node10
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH -c 12
#SBATCH -t 0-04:00:00
#SBATCH -J tuneglim
#SBATCH -o /home/%u/abstraction-grader/runs/_batch/slurm-%j-tuneglimmer.out
# Muse Glimmer-30B tuning (condition B): the 3 tuning queries x low/medium/xhigh, seed 0 (9 runs).
# Round 0 (base prompts, with smoke test):  sbatch jobs/tune_glimmer.sbatch
# Round k:                                   sbatch --export=ALL,PROMPTS=glimmer_r1,SKIP_SMOKE=1 jobs/tune_glimmer.sbatch
MODEL_KEY=muse-glimmer-30b HF_ID=meta-models/Muse-Glimmer-30B PORT=8003 SEEDS=0
TAG=tune-${PROMPTS:-glimmer_r0}${TAG_SUFFIX:-} QUERIES=stockindex:1,bookreview:1,yelp:1 SKIP_SMOKE=${SKIP_SMOKE:-0} HF_EXCLUDE=''
SERVE_ARGS="--tensor-parallel-size 2 --language-model-only --generation-config auto --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer"
source ~/abstraction-grader/jobs/_common.sh
