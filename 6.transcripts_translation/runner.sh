#!/bin/bash
#SBATCH --job-name=lt_infer_align
#SBATCH --mail-type="BEGIN,END,FAIL,STAGE_OUT,ARRAY_TASKS"
#SBATCH --mail-user=slack:U044ZUGHAQN
# --output=slurm_out/slurm-%x.%j.out
# --error=slurm_out/slurm-%x.%j.err
# --output=slurm_out/slurm-%x.%a.%A_%j.out
# --error=slurm_out/slurm-%x.%a.%A_%j.err
# --output=slurm_out/slurm-%x.%j.log
#SBATCH --output=slurm_out/slurm-%x.%a.%A_%j.log
#SBATCH --exclude="elm[11-15],elm[21-26],elm[41-44],elm[51-55]"

#python runner.py --model "gemma3it12b" --dummy


if [ "$SLURM_ARRAY_TASK_ID" -eq 666 ]; then
    exit 1

elif [ "$SLURM_ARRAY_TASK_ID" -eq 993 ]; then
    python runner.py --model "aya_vision_8b" --context "" --dummy --use-non-corrected-asr
    python runner.py --model "aya_vision_8b" --context "slidetxt" --dummy --use-non-corrected-asr
    python runner.py --model "aya_vision_8b" --context "slideimg" --dummy --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 994 ]; then
    python runner.py --model "glm46v_flash" --context "" --dummy --use-non-corrected-asr
    python runner.py --model "glm46v_flash" --context "slidetxt" --dummy --use-non-corrected-asr
    python runner.py --model "glm46v_flash" --context "slideimg" --dummy --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 995 ]; then
    python runner.py --model "internvl35_8b" --context "" --dummy --use-non-corrected-asr
    python runner.py --model "internvl35_8b" --context "slidetxt" --dummy --use-non-corrected-asr
    python runner.py --model "internvl35_8b" --context "slideimg" --dummy --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 996 ]; then
    python runner.py --model "qwen35_9b" --context "" --dummy --use-non-corrected-asr
    python runner.py --model "qwen35_9b" --context "slidetxt" --dummy --use-non-corrected-asr
    python runner.py --model "qwen35_9b" --context "slideimg" --dummy --use-non-corrected-asr
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


elif [ "$SLURM_ARRAY_TASK_ID" -eq 31 ]; then
    python runner.py --model "qwen35_9b" --context ""
elif [ "$SLURM_ARRAY_TASK_ID" -eq 32 ]; then
    python runner.py --model "qwen35_9b" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 33 ]; then
    python runner.py --model "qwen35_9b" --context "slideimg"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 41 ]; then
    python runner.py --model "aya_vision_8b" --context ""
elif [ "$SLURM_ARRAY_TASK_ID" -eq 42 ]; then
    python runner.py --model "aya_vision_8b" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 43 ]; then
    python runner.py --model "aya_vision_8b" --context "slideimg"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 251 ]; then
    python runner.py --model "glm46v_flash" --context ""
elif [ "$SLURM_ARRAY_TASK_ID" -eq 252 ]; then
    python runner.py --model "glm46v_flash" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 253 ]; then
    python runner.py --model "glm46v_flash" --context "slideimg"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 261 ]; then
    python runner.py --model "internvl35_8b" --context ""
elif [ "$SLURM_ARRAY_TASK_ID" -eq 262 ]; then
    python runner.py --model "internvl35_8b" --context "slidetxt"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 263 ]; then
    python runner.py --model "internvl35_8b" --context "slideimg"


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


elif [ "$SLURM_ARRAY_TASK_ID" -eq 36 ]; then
    python runner.py --model "qwen35_9b" --context "" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 37 ]; then
    python runner.py --model "qwen35_9b" --context "slidetxt"  --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 38 ]; then
    python runner.py --model "qwen35_9b" --context "slideimg"  --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 46 ]; then
    python runner.py --model "aya_vision_8b" --context "" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 47 ]; then
    python runner.py --model "aya_vision_8b" --context "slidetxt"  --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 48 ]; then
    python runner.py --model "aya_vision_8b" --context "slideimg"  --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 56 ]; then
    python runner.py --model "glm46v_flash" --context "" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 57 ]; then
    python runner.py --model "glm46v_flash" --context "slidetxt"  --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 58 ]; then
    python runner.py --model "glm46v_flash" --context "slideimg"  --use-non-corrected-asr

elif [ "$SLURM_ARRAY_TASK_ID" -eq 66 ]; then
    python runner.py --model "internvl35_8b" --context "" --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 67 ]; then
    python runner.py --model "internvl35_8b" --context "slidetxt"  --use-non-corrected-asr
elif [ "$SLURM_ARRAY_TASK_ID" -eq 68 ]; then
    python runner.py --model "internvl35_8b" --context "slideimg"  --use-non-corrected-asr




#####################################
#  Slide Task
#####################################


elif [ "$SLURM_ARRAY_TASK_ID" -eq 101 ]; then
    python runner.py --model "llama32" --task "slide"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 103 ]; then
    python runner.py --model "llama32" --context "slideimg" --task "slide"


# python runner.py --model "llama4scout"



elif [ "$SLURM_ARRAY_TASK_ID" -eq 111 ]; then
    python runner.py --model "gemma3it12b" --task "slide"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 113 ]; then
    python runner.py --model "gemma3it12b" --context "slideimg" --task "slide"





#####################################
#  VQA
#####################################


elif [ "$SLURM_ARRAY_TASK_ID" -eq 520 ]; then
    python runner.py --model "llama4scout" --context "vqa"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 521 ]; then
    python runner.py --model "llama4scout" --context "vqa"  --range 11 99


elif [ "$SLURM_ARRAY_TASK_ID" -eq 530 ]; then
    python runner.py --model "qwen35_9b" --context "vqa"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 531 ]; then
    python runner.py --model "qwen35_9b" --context "vqa"  --range 11 99


elif [ "$SLURM_ARRAY_TASK_ID" -eq 550 ]; then
    python runner.py --model "glm46v_flash" --context "vqa"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 551 ]; then
    python runner.py --model "glm46v_flash" --context "vqa"  --range 11 99



fi
