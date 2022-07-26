#ifndef GEN_H
#define GEN_H

#include "bb.h"
#include "board.h"
#include "move.h"

__device__ __host__ int is_check(Board *board, char color);
__device__ __host__ int is_illegal(Board *board);
__device__ __host__ int gen_moves(Board *board, Move *moves);
//__global__ void gen_moves_gpu(Board *board, Move *moves, bb *dsts_array, int *pieces, int *count);
__global__ void gen_moves_gpu(Board *board_arr, Move *moves_arr, int *count_arr, int childIdx);
int gen_legal_moves(Board *board, Move *moves);

#endif
