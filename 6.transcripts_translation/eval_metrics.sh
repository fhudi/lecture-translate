#!/bin/bash
#SBATCH --job-name=lt_eval
#--mail-type="BEGIN,END,FAIL,STAGE_OUT,ARRAY_TASKS"
#--mail-user=slack:U044ZUGHAQN
# --output=slurm_out/slurm-%x.%j.out
# --error=slurm_out/slurm-%x.%j.err
# --output=slurm_out/slurm-%x.%a.%A_%j.out
# --error=slurm_out/slurm-%x.%a.%A_%j.err
# --output=slurm_out/slurm-%x.%j.log
#SBATCH --output=slurm_out/slurm-%x.%a.%A_%j.log
#--exclude="elm[11-15],elm[21-26],elm[41-44],elm[51-55]"


# export __EVAL_USE_COMET__=0
# export __EVAL_SUFFIX_NAME__="transcripts_annotd"
# export __EVAL_SUFFIX_NAME__="transcripts_aligned"


if [ "$SLURM_ARRAY_TASK_ID" -eq 666 ]; then
    exit 1

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    export __EVAL_SPLIT_ST__=0
    export __EVAL_SPLIT_ED__=3
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    export __EVAL_SPLIT_ST__=3
    export __EVAL_SPLIT_ED__=6
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    export __EVAL_SPLIT_ST__=6
    export __EVAL_SPLIT_ED__=9
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    export __EVAL_SPLIT_ST__=9
    export __EVAL_SPLIT_ED__=12
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    export __EVAL_SPLIT_ST__=12
    export __EVAL_SPLIT_ED__=15
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    export __EVAL_SPLIT_ST__=15
    export __EVAL_SPLIT_ED__=18
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
    export __EVAL_SPLIT_ST__=18
    export __EVAL_SPLIT_ED__=21
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
    export __EVAL_SPLIT_ST__=21
    export __EVAL_SPLIT_ED__=24
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
    export __EVAL_SPLIT_ST__=24
    export __EVAL_SPLIT_ED__=27
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    export __EVAL_SPLIT_ST__=27
    export __EVAL_SPLIT_ED__=30
    python eval_metrics.py


elif [ "$SLURM_ARRAY_TASK_ID" -eq 101 ]; then
    export __EVAL_SPLIT_ST__=0
    export __EVAL_SPLIT_ED__=2
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 102 ]; then
    export __EVAL_SPLIT_ST__=2
    export __EVAL_SPLIT_ED__=4
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 103 ]; then
    export __EVAL_SPLIT_ST__=4
    export __EVAL_SPLIT_ED__=6
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 104 ]; then
    export __EVAL_SPLIT_ST__=6
    export __EVAL_SPLIT_ED__=8
    python eval_metrics.py

elif [ "$SLURM_ARRAY_TASK_ID" -eq 105 ]; then
    export __EVAL_SPLIT_ST__=8
    export __EVAL_SPLIT_ED__=10
    python eval_metrics.py


fi
