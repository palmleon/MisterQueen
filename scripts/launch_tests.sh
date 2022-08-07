#!/bin/bash

SRC_PATH="src"
EXEC_PATH="bin/release/main"
LOG_DIR="timings"

for depth in {4..10}; do

    # set the new MAX_DEPTH
    sed -e "7s/[0-9]* *$/$depth/" -i $SRC_PATH/config.h
        
    for thrnode in 8 16; do

        # set the new THREADS_PER_NODE
        sed -e "6s/[0-9]* *$/$thrnode/" -i $SRC_PATH/config.h
        # set the log file name
        LOG_FILE="inlining_depth${depth}_thrnode${thrnode}.log"
        # recompile everything
        make 
        # launch tests
        exec $EXEC_PATH > $LOG_DIR/$LOG_FILE

    done
done