#!/bin/bash
#SBATCH --job-name=ll
#SBATCH --mail-type="BEGIN,END,FAIL,STAGE_OUT,ARRAY_TASKS"
#SBATCH --mail-user=slack:U044ZUGHAQN
# --output=slurm_out/slurm-%x.%j.out
# --error=slurm_out/slurm-%x.%j.err
#SBATCH --output=slurm_out/slurm-%x.%A_%a.%j.out
#SBATCH --error=slurm_out/slurm-%x.%A_%a.%j.err


if [ "$SLURM_ARRAY_TASK_ID" -eq 666 ]; then
    exit 1

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
#    uv run runner.py --model "llama32"
    python runner.py --model "llama32"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    python runner.py --model "llama32" --context "slidetxt"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    python runner.py --model "llama32" --context "slideimg"

# python runner.py --model "llama4scout"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 11 ]; then
    python runner.py --model "gemma3it"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then
    python runner.py --model "gemma3it" --context "slidetxt"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 13 ]; then
    python runner.py --model "gemma3it" --context "slideimg"


fi
