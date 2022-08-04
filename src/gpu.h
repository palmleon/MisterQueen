#ifndef GPU_H
#define GPU_H

#include "board.h"
#include "move.h"

__global__ void alpha_beta_gpu_kernel(Board *board, Move *first_moves, Move *second_moves, int *pos_parent, int *scores, int depth, int alpha, int beta, int count, int baseIdx);
__global__ void alpha_beta_gpu_kernel(Board *board, Move *first_moves, int depth, int alpha, int beta, int *scores);
__global__ void alpha_beta_gpu_kernel(Board *board, int depth, int alpha, int beta, int *scores);

#endif