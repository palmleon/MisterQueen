#ifndef EVAL_H
#define EVAL_H

#include "board.h"
#include "device_launch_parameters.h"

__device__ __host__ int evaluate(Board *board);
__device__ __host__ int evaluate_pawns(Board *board);

#endif
