#!/bin/bash

SRC_PATH="src"
EXEC_PATH="bin/release/main"
LOG_DIR="timings"

for depth in {4..10}; do

    # set the new MAX_DEPTH
    sed -e "7s/[0-9]* *$/$depth/" -i $SRC_PATH/config.h
    # set the log file name
    LOG_FILE="sequential_depth${depth}.log"
    # recompile everything
    make 
    # launch tests
    echo -e "\033[0;32mLaunching tests!\033[0m"
    ./bin/release/main > $LOG_DIR/$LOG_FILE
done