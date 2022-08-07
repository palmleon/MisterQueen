#!/bin/bash

SRC_PATH="src"
EXEC_PATH="bin/release/main"
LOG_DIR="timings"

for depth_tot in {4..10}; do

    for d in {1..3}; do

        for thrnode in 16 32; do 

            (( s = depth_tot - d ))

            # set the new MAX_DEPTH_SEQ
            sed -e "10s/[0-9]* *$/$s/" -i $SRC_PATH/config.h

            # set the new MAX_DEPTH_PAR
            sed -e "11s/[0-9]* *$/$d/" -i $SRC_PATH/config.h

            # set the new THREADS_PER_NODE
            sed -e "5s/[0-9]* *$/$thrnode/" -i $SRC_PATH/config.h

            #set the log file name
            LOG_FILE="rockisuda_s${s}d${d}_thrnode${thrnode}.log"

            #recompile everything
            make

            #launch tests
            echo -e "\033[0;32mLaunching tests!\033[0m"
            exec $EXEC_PATH > $LOG_DIR/$LOG_FILE

        done

    done

done
