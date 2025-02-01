#!/bin/bash

set -e
TOP_DIR=$PWD
TOOLCHAIN_FILE=?
SDK_DIR=?


rm -rf build
mkdir -p build
case "$1" in
"sdk")
    cd $SDK_DIR
    source envsetup.sh
    cd $TOP_DIR/build

    cmake -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE $TOP_DIR
    ;;
"pc")
    cd $TOP_DIR/build

    cmake $TOP_DIR
    ;;
esac
make -j6 install
