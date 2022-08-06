#ifndef GEN_H
#define GEN_H

#include "bb.h"
#include "board.h"
#include "move.h"

#define THREADS_PER_NODE 64

__device__ __host__ int is_check(Board *board, char color);
__device__ __host__ int is_illegal(Board *board);
int gen_moves(Board *board, Move *moves);
__global__ void gen_moves_gpu(Board *board_arr, Move *moves_arr, int *count_arr, int baseNodeIdx, int nNodes);
int gen_legal_moves(Board *board, Move *moves);

#endif
