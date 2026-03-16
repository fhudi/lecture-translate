#!/bin/bash
#SBATCH --job-name=lectrans_infer
#SBATCH --mail-type="BEGIN,END,FAIL,STAGE_OUT,ARRAY_TASKS"
#SBATCH --mail-user=slack:U044ZUGHAQN
# --output=slurm_out/slurm-%x.%j.out
# --error=slurm_out/slurm-%x.%j.err
# --output=slurm_out/slurm-%x.%a.%A_%j.out
# --error=slurm_out/slurm-%x.%a.%A_%j.err
# --output=slurm_out/slurm-%x.%j.log
#SBATCH --output=slurm_out/slurm-%x.%a.%A_%j.log

#python runner.py --model "gemma3it12b" --dummy


if [ "$SLURM_ARRAY_TASK_ID" -eq 666 ]; then
    exit 1

elif [ "$SLURM_ARRAY_TASK_ID" -eq 997 ]; then
    python runner.py --model "gemma3it12b" --context "" --dummy --use-non-corrected-asr
    python runner.py --model "gemma3it12b" --context "slidetxt" --dummy --use-non-corrected-asr
    python runner.py --model "gemma3it12b" --context "slideimg" --dummy --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 998 ]; then
    python runner.py --model "llama32"     --context "" --dummy --use-non-corrected-asr
    python runner.py --model "llama32"     --context "slidetxt" --dummy --use-non-corrected-asr
    python runner.py --model "llama32"     --context "slideimg" --dummy --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 999 ]; then
    python runner.py --model "llama4scout" --context "" --dummy --use-non-corrected-asr
    python runner.py --model "llama4scout" --context "slideimg" --dummy --use-non-corrected-asr


###################################
#  Corrected ASR output as input  #
###################################

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
#    uv run runner.py --model "llama32"
    python runner.py --model "llama32"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    python runner.py --model "llama32" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    python runner.py --model "llama32" --context "slideimg"


elif [ "$SLURM_ARRAY_TASK_ID" -eq 21 ]; then
    python runner.py --model "llama4scout"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 22 ]; then
    python runner.py --model "llama4scout" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 23 ]; then
    python runner.py --model "llama4scout" --context "slideimg"  # --range 11 99

elif [ "$SLURM_ARRAY_TASK_ID" -eq 11 ]; then
    python runner.py --model "gemma3it12b"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then
    python runner.py --model "gemma3it12b" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 13 ]; then
    python runner.py --model "gemma3it12b" --context "slideimg"



#####################################
#  ASR output as transcript source  #
#####################################

elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    python runner.py --model "llama32" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
    python runner.py --model "llama32" --context "slidetxt" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
    python runner.py --model "llama32" --context "slideimg" --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 26 ]; then
    python runner.py --model "llama4scout" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 27 ]; then
    python runner.py --model "llama4scout" --context "slidetxt" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 28 ]; then
    python runner.py --model "llama4scout" --context "slideimg" --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 16 ]; then
    python runner.py --model "gemma3it12b" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 17 ]; then
    python runner.py --model "gemma3it12b" --context "slidetxt" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 18 ]; then
    python runner.py --model "gemma3it12b" --context "slideimg" --use-non-corrected-asr






elif [ "$SLURM_ARRAY_TASK_ID" -eq 101 ]; then
    python runner.py --model "llama32" --task "slide"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 103 ]; then
    python runner.py --model "llama32" --context "slideimg" --task "slide"


# python runner.py --model "llama4scout"



elif [ "$SLURM_ARRAY_TASK_ID" -eq 111 ]; then
    python runner.py --model "gemma3it12b" --task "slide"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 113 ]; then
    python runner.py --model "gemma3it12b" --context "slideimg" --task "slide"


fi
