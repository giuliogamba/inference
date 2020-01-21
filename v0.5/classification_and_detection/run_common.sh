#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 tf|onnxruntime|pytorch|tflite|pynq [resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|lfcW1A1|lfcW1A2|cnvW1A1|cnvW1A2|cnvW2A2] [cpu|gpu|fpga]"
    exit 1
fi
if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi

# defaults
backend=tf
model=resnet50
device="cpu"

for i in $* ; do
    case $i in
       tf|onnxruntime|tflite|pytorch|pynq) backend=$i; shift;;
       cpu|gpu|fpga) device=$i; shift;;
       gpu) device=gpu; shift;;
       resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|ssd-resnet34-tf|lfcW1A1|lfcW1A2|cnvW1A1|cnvW1A2|cnvW2A2) model=$i; shift;;
    esac
done

if [[ ($device == "cpu") || ($device == "fpga") ]] ; then
    export CUDA_VISIBLE_DEVICES=""
fi

if [[ ("$BOARD" == "Pynq-Z1") || ("$BOARD" == "Pynq-Z2") ]]; then
  PLATFORM="pynqZ1-Z2"  
elif [[ ("$BOARD" == "Ultra96") ]]; then
  PLATFORM="ultra96"
fi

name="$model-$backend"
extra_args=""


#
# pynq
#
if [ $name == "lfcW1A1-pynq" ] ; then
    model_path="$MODEL_DIR/$PLATFORM/$model-$PLATFORM.bit"
    profile=lfcW1A1-pynq
    extra_args="$extra_args --backend pynq"
fi
if [ $name == "lfcW1A2-pynq" ] ; then
    model_path="$MODEL_DIR/$PLATFORM/$model-$PLATFORM.bit"
    profile=lfcW1A2-pynq
    extra_args="$extra_args --backend pynq"
fi
if [ $name == "cnvW1A1-pynq" ] ; then
    model_path="$MODEL_DIR/$PLATFORM/$model-$PLATFORM.bit"
    profile=cnvW1A1-pynq
    extra_args="$extra_args --backend pynq"
fi
if [ $name == "cnvW1A2-pynq" ] ; then
    model_path="$MODEL_DIR/$PLATFORM/$model-$PLATFORM.bit"
    profile=cnvW1A2-pynq
    extra_args="$extra_args --backend pynq"
fi
if [ $name == "cnvW2A2-pynq" ] ; then
    model_path="$MODEL_DIR/$PLATFORM/$model-$PLATFORM.bit"
    profile=cnvW2A2-pynq
    extra_args="$extra_args --backend pynq"
fi
if [ $name == "resnet50-pynq" ] ; then
    model_path="$MODEL_DIR/resnet50.xclbin"
    profile=resnet50-pynq
    extra_args="$extra_args --backend pynq"
fi
#
# tensorflow
#
if [ $name == "resnet50-tf" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.pb"
    profile=resnet50-tf
fi
if [ $name == "mobilenet-tf" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224_frozen.pb"
    profile=mobilenet-tf
fi
if [ $name == "ssd-mobilenet-tf" ] ; then
    model_path="$MODEL_DIR/ssd_mobilenet_v1_coco_2018_01_28.pb"
    profile=ssd-mobilenet-tf
fi
if [ $name == "ssd-resnet34-tf" ] ; then
    model_path="$MODEL_DIR/resnet34_tf.22.1.pb"
    profile=ssd-resnet34-tf
fi

#
# onnxruntime
#
if [ $name == "resnet50-onnxruntime" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.onnx"
    profile=resnet50-onnxruntime
fi
if [ $name == "mobilenet-onnxruntime" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224.onnx"
    profile=mobilenet-onnxruntime
fi
if [ $name == "ssd-mobilenet-onnxruntime" ] ; then
    model_path="$MODEL_DIR/ssd_mobilenet_v1_coco_2018_01_28.onnx"
    profile=ssd-mobilenet-onnxruntime
fi
if [ $name == "ssd-resnet34-onnxruntime" ] ; then
    # use onnx model converted from pytorch
    model_path="$MODEL_DIR/resnet34-ssd1200.onnx"
    profile=ssd-resnet34-onnxruntime
fi
if [ $name == "ssd-resnet34-tf-onnxruntime" ] ; then
    # use onnx model converted from tensorflow
    model_path="$MODEL_DIR/ssd_resnet34_mAP_20.2.onnx"
    profile=ssd-resnet34-onnxruntime-tf
fi

#
# pytorch
#
if [ $name == "resnet50-pytorch" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.onnx"
    profile=resnet50-onnxruntime
    extra_args="$extra_args --backend pytorch"
fi
if [ $name == "mobilenet-pytorch" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224.onnx"
    profile=mobilenet-onnxruntime
    extra_args="$extra_args --backend pytorch"
fi
if [ $name == "ssd-resnet34-pytorch" ] ; then
    model_path="$MODEL_DIR/resnet34-ssd1200.pytorch"
    profile=ssd-resnet34-pytorch
fi


#
# tflite
#
if [ $name == "resnet50-tflite" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.tflite"
    profile=resnet50-tf
    extra_args="$extra_args --backend tflite"
fi
if [ $name == "mobilenet-tflite" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224.tflite"
    profile=mobilenet-tf
    extra_args="$extra_args --backend tflite"
fi

name="$backend-$device/$model"
EXTRA_OPS="$extra_args $EXTRA_OPS"
