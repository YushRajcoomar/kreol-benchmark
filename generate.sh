#!/bin/bash

BASEDIR=$(pwd)
FAIRSEQ=${BASEDIR}/fairseq/fairseq_cli
SRC_LANG='en'
TGT_LANG='cr'
DATA=${BASEDIR}/postprocessed/${NAME}/${SRC_LANG}-${TGT_LANG}
langs=${BASEDIR}/mbart50.ft.nn/ML50_langs.txt

RESULTS_PATH=${BASEDIR}/results

MODEL_PATH=${BASEDIR}/checkpoint/checkpoint_last.pt

BPE_TYPE='sentencepiece'
SPM_MODEL=${BASEDIR}/mbart50.ft.nn/sentence.bpe.model

CUDA_VISIBLE_DEVICES=0,1 fairseq-generate ${DATA} --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} \
  --path $MODEL_PATH \
  --results-path $RESULTS_PATH \
  --bpe $BPE_TYPE --sentencepiece-model $SPM_MODEL \
  --task translation_from_pretrained_bart \
  --langs $langs \
  --nbest 3 \
  --beam 5 \
	--batch-size 5 --sacrebleu > en_cr 


# cat en_cr | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[cr\]//g' |$TOKENIZER cr > en_cr.hyp
# cat en_cr | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[cr\]//g' |$TOKENIZER cr > en_cr.ref
# sacrebleu -tok 'none' -s 'none' en_cr.ref < en_cr.hyp