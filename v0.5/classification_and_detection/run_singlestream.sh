#!/bin/bash

source ./run_common.sh

# for single stream its 1 image per query
BS=1
COUNT=$(( $BS*1024 )) 
SPQ=$BS
common_opt="--config ../mlperf.conf"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=`pwd`/output_SingleStream_BATCH-SIZE_${BS}_COUNT_${COUNT}/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# command for resnet50 SingleStream
python3 python/main.py --profile $profile $common_opt --model $model_path $dataset \
    --output $OUTPUT_DIR $EXTRA_OPS --scenario SingleStream --accuracy --count $COUNT \
    --max-batchsize $BS --samples-per-query $SPQ $@

# python3 -m cProfile -o temp.cprof python/main.py --profile $profile $common_opt --model $model_path $dataset \
#     --output $OUTPUT_DIR $EXTRA_OPS --scenario SingleStream --threads 4 --count 10000 --max-batchsize 10000 --accuracy $@

# pyprof2calltree -k -i temp.cprof