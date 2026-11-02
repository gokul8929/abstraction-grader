#!/bin/bash
# Submit the three jobs one after another on node10 (each waits for the previous one to end, success or not).
set -e
cd ~/abstraction-grader/jobs
j1=$(sbatch --parsable gptoss.sbatch)
j2=$(sbatch --parsable --dependency=afterany:$j1 glm.sbatch)
j3=$(sbatch --parsable --dependency=afterany:$j2 qwen_seed1.sbatch)
echo "submitted: gpt-oss $j1 -> GLM $j2 -> Qwen seed 1 $j3"
squeue -u $USER
