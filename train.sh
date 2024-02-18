BASEDIR=$(pwd)
FAIRSEQ=${BASEDIR}/fairseq/fairseq_cli
PRETRAIN=${BASEDIR}/mbart50.ft.nn/
langs=${BASEDIR}/mbart50.ft.nn/ML50_langs.txt
SRC=en
TGT=cr
NAME=en-cr
DATADIR=${BASEDIR}/postprocessed/${NAME}
SAVEDIR=${BASEDIR}/checkpoint

python ${FAIRSEQ}/train.py ${DATADIR}  --encoder-normalize-before --decoder-normalize-before  --arch mbart_large \
--task translation_from_pretrained_bart  --source-lang ${SRC} --target-lang ${TGT} --criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 --dropout 0.3 \
--attention-dropout 0.1 --weight-decay 0.0 --max-tokens 768 --update-freq 2 --save-interval 1 --save-interval-updates 8000 \
--keep-interval-updates 10 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 100 --reset-optimizer \
--reset-meters --reset-dataloader --save-dir ${SAVEDIR} --reset-lr-scheduler --restore-file $PRETRAIN --langs $langs --layernorm-embedding \
--ddp-backend no_c10d 
