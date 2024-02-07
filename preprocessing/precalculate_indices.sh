BASEDIR="C:\Users\yush2\OneDrive\Desktop\kreol"
DATA=${BASEDIR}\\experiments\\data
FAIRSEQ=${BASEDIR}\\fairseq\\fairseq_cli
SRC=en
TGT=cr
NAME=en-cr
TRAIN=train
VALID=dev
TEST=test
DEST=${BASEDIR}\\postprocessed
DICT=${BASEDIR}\\preprocessing\\dict.en_fr_cr.txt

python ${FAIRSEQ}\\preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}\\${NAME}\\indices\\${NAME}_${TRAIN}.spm \
--validpref ${DATA}\\${NAME}\\indices\\${NAME}_${VALID}.spm \
--testpref ${DATA}\\${NAME}\\indices\\${NAME}_${TEST}.spm  \
--destdir ${DEST}\\${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 1