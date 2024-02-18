
BASEDIR=$(pwd)
FAIRSEQ=${BASEDIR}/fairseq/fairseq_cli
SRC_LANG='en'
TGT_LANG='cr'
DATA=${BASEDIR}/postprocessed/${NAME}/${SRC_LANG}-${TGT_LANG}
langs=${BASEDIR}/mbart50.ft.nn/ML50_langs.txt
MODEL_PATH=${BASEDIR}/checkpoint/checkpoint_last.pt

fairseq-eval-lm ${DATA} --path ${MODEL_PATH} \
    --sample-break-mode complete \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400