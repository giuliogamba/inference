#!/bin/bash

source ./run_common.sh


# 2000 batch size is because it divides the dataset of 50000 equally
BS=2000
COUNT=50000 
SPQ=$BS
common_opt="--config ../mlperf.conf"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=`pwd`/output_Offline_BATCH-SIZE_${BS}_COUNT_${COUNT}/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# command for resnet50 SingleStream
python3 python/main.py --profile $profile $common_opt --model $model_path $dataset \
    --output $OUTPUT_DIR $EXTRA_OPS --scenario Offline --accuracy --count $COUNT \
    --max-batchsize $BS --threads 16 $@

# python3 -m cProfile -o temp.cprof python/main.py --profile $profile $common_opt --model $model_path $dataset \
#     --output $OUTPUT_DIR $EXTRA_OPS --scenario SingleStream --threads 4 --count 10000 --max-batchsize 10000 --accuracy $@

# pyprof2calltree -k -i temp.cprof