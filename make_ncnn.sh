#!/bin/bash

set -e
TOP_DIR=$PWD
NCNN_DIR=?
TOOLCHAIN_FILE=?
SDK_DIR=?

rm -rf build
mkdir -p build
case "$1" in
"sdk")
    cd $SDK_DIR
    source envsetup.sh
    cd $TOP_DIR/build

    cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE \
        -DNCNN_C_API=OFF \
        -DNCNN_SIMPLEVK=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF \
        $NCNN_DIR
    ;;
"pc")
    cd $TOP_DIR/build
    cmake \
        -DNCNN_C_API=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF \
        $NCNN_DIR
    ;;
esac

make -j6 install

mkdir -p $TOP_DIR/lib/build/"$1"/ncnn/
cp -r $TOP_DIR/build/install/* $TOP_DIR/lib/build/"$1"/ncnn/