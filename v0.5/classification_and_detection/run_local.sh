#!/bin/bash

source ./run_common.sh

common_opt="--config ../mlperf.conf"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=`pwd`/output/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi


python3 python/main.py --profile $profile $common_opt --model $model_path $dataset \
    --output $OUTPUT_DIR $EXTRA_OPS --scenario SingleStream --threads 4 --count 10000 --max-batchsize 10000 --accuracy $@

# python3 -m cProfile -o temp.cprof python/main.py --profile $profile $common_opt --model $model_path $dataset \
#     --output $OUTPUT_DIR $EXTRA_OPS --scenario SingleStream --threads 4 --count 10000 --max-batchsize 10000 --accuracy $@

