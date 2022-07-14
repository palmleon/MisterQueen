#ifndef GPU_H
#define GPU_H

#include "board.h"

__device__ __forceinline__ void alpha_beta_gpu_iter(Board *board_parent, int depth, int alpha_parent, int beta_parent, Move* moves_parent, int* scores_parent);
__global__ void alpha_beta_gpu_kernel(Board *board_parent, int depth, int alpha, int beta, Move* moves_parent, int* scores);

#endif