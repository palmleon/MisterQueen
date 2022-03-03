#include "search.cuh"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include "eval.cuh"
#include "gen.cuh"
#include "move.cuh"
#include "util.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define XOR_SWAP(a, b) a = a ^ b; b = a ^ b; a = a ^ b;

__device__ __host__ void sort_moves(Board *board, Move *moves, int count) {
    int scores[MAX_MOVES];
    int indexes[MAX_MOVES];
    for (int i = 0; i < count; i++) {
        Move *move = moves + i;
        scores[i] = score_move(board, move);
        indexes[i] = i;
    }
    for (int i = 1; i < count; i++) {
        int j = i;
        while (j > 0 && scores[j - 1] < scores[j]) {
            XOR_SWAP(scores[j - 1], scores[j]);
            XOR_SWAP(indexes[j - 1], indexes[j]);
            j--;
        }
    }
    Move temp[MAX_MOVES];
    memcpy(temp, moves, sizeof(Move) * count);
    for (int i = 0; i < count; i++) {
        memcpy(moves + i, temp + indexes[i], sizeof(Move));
    }
}

__device__ int alpha_beta(Search *search, Board *board, int depth, int ply, int alpha, int beta) {
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board) + evaluate_pawns(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int count = gen_moves(board, moves);
        sort_moves(board, moves, count);
        int can_move = 0;
        for (int i = 0; i < count; i++) {
            Move *move = &moves[i];
            do_move(board, move, &undo);
            int score = -alpha_beta(search, board, depth - 1, ply + 1, -beta, -alpha);
            undo_move(board, move, &undo);
            if (score > -INF) {
                can_move = 1;
            }
            if (score >= beta) {
                return beta;
            }
            if (score > alpha) {
                alpha = score;
            }
        }
        result = alpha;
        if (!can_move) {
            if (is_check(board)) {
                result = -MATE + ply;
            } else {
                result = 0;
            }
        }
    }
    return result;
}

__global__ void GPU_alpha_beta(Search *search, Board *board, Move *moves, int depth, int ply, int alpha, int beta, int *scores) {
    
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);
	int local_index = threadIdx.x;
    
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board) + evaluate_pawns(board);
    }
    else {
        Undo undo;
        int can_move = 0;
        do_move(board, &moves[local_index], &undo);
        scores[local_index] = -alpha_beta(search, board, depth - 1, ply + 1, -beta, -alpha);
        undo_move(board, &moves[local_index], &undo);
    }
    return;
}

int CPU_alpha_beta(Search *search, Board *board, int depth, int ply, int alpha, int beta) { 
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board) + evaluate_pawns(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int *scores = NULL; 

        Search *dev_search = NULL;
	    Board *dev_board = NULL;
	    Move *dev_moves = NULL;
        int *dev_scores = NULL;
	    cudaError_t cudaStatus;

        int count = gen_moves(board, moves);
        sort_moves(board, moves, count);
        int can_move = 0; 
        do_move(board, &moves[0], &undo);
        int score = -CPU_alpha_beta(search, board, depth - 1, ply + 1, -beta, -alpha);
        undo_move(board, &moves[0], &undo);
        if (score > -INF) {
            can_move = 1;
        }
        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }

        scores = (int *) malloc(count-1 * sizeof(int));
        if (scores == NULL) {
            fprintf(stderr, "malloc failed!");
            alpha = -1;
            goto Error;
        }

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            alpha = -1;
            goto Error;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_search, sizeof(Search));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            alpha = -1;
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_board, sizeof(Board));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            alpha = -1;
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_moves, count-1 * sizeof(Move));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            alpha = -1;
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_scores, count-1 * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            alpha = -1;
            goto Error;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_search, search, sizeof(Search), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            alpha = -1;
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_board, board, sizeof(Board), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            alpha = -1;
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_moves, moves+1, count-1 * sizeof(Move), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            alpha = -1;
            goto Error;
        }

        
        GPU_alpha_beta <<<1, count-1>>> (dev_search, dev_board, moves, depth -1, ply +1, -beta, -alpha, dev_scores);
        
        cudaStatus = cudaMemcpy(scores, dev_scores, count-1 * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            alpha = -1;
            goto Error;
        }

        for (int i = 1; i < count; i++) {
            if (scores[i] > -INF) {
                can_move = 1;
            }
            if (scores[i] >= beta) {
                return beta;
            }
            if (scores[i] > alpha) {
                alpha = score;
            }
        }
        result = alpha;
        if (!can_move) {
            if (is_check(board)) {
                result = -MATE + ply;
            } else {
                result = 0;
            }
        }
    Error:
        free(scores);
        cudaFree(dev_search);
        cudaFree(dev_board);
        cudaFree(dev_moves);
        cudaFree(dev_scores);
    }
    return result;
}

int root_search(Search *search, Board *board, int depth, int ply, int alpha, int beta, Move *result) {
    Undo undo;
    Move moves[MAX_MOVES];
    int count = gen_moves(board, moves);
    sort_moves(board, moves, count);
    Move *best = NULL;
    int *scores = NULL; 

    Search *dev_search = NULL;
	Board *dev_board = NULL;
	Move *dev_moves = NULL;
    int *dev_scores = NULL;
	cudaError_t cudaStatus;

    do_move(board, &moves[0], &undo);
    int score = -CPU_alpha_beta(search, board, depth - 1, ply + 1, -beta, -alpha);
    undo_move(board, &moves[0], &undo);
    if (score > alpha) {
        alpha = score;
        *best = moves[0];
    }

    scores = (int *) malloc(count-1 * sizeof(int));
    if (scores == NULL) {
        fprintf(stderr, "malloc failed!");
		alpha = -1;
		goto Error;
    }

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        alpha = -1;
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_search, sizeof(Search));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		alpha = -1;
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_board, sizeof(Board));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		alpha = -1;
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_moves, count-1 * sizeof(Move));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		alpha = -1;
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)&dev_scores, count-1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		alpha = -1;
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_search, search, sizeof(Search), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		alpha = -1;
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_board, board, sizeof(Board), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		alpha = -1;
		goto Error;
	}

    cudaStatus = cudaMemcpy(dev_moves, moves+1, count-1 * sizeof(Move), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		alpha = -1;
		goto Error;
	}

    
    GPU_alpha_beta <<<1, count-1>>> (dev_search, dev_board, moves, depth -1, ply +1, -beta, -alpha, dev_scores);
    
    cudaStatus = cudaMemcpy(scores, dev_scores, count-1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		alpha = -1;
		goto Error;
	}

    for (int i = 1; i < count; i++) {
        if (scores[i] > alpha) {
            alpha = scores[i];
            *best = moves[i];
        }
    }
    if (best) {
        memcpy(result, best, sizeof(Move));
    }

Error:
    free(scores);
	cudaFree(dev_search);
	cudaFree(dev_board);
	cudaFree(dev_moves);
    cudaFree(dev_scores);

	return alpha;
}

int pv_split(Search *search, Board *board, int depth, int ply, int alpha, int beta, int length, Move *result) {
    if (length <= 0){
        //return treesplit(search, board, depth, ply, alpha, beta, result) KERNEL LAUNCH is inside this function
    }
    Undo undo;
    Move moves[MAX_MOVES];
    int count = gen_moves(board, moves); // may be parallelized
    sort_moves(board, moves, count);    // may be parallelized

    Move *best = NULL;
    int *scores = NULL; 
    Search *dev_search = NULL;
	Board *dev_board = NULL;
	Move *dev_moves = NULL;
    int *dev_scores = NULL;
	cudaError_t cudaStatus;

    do_move(board, &moves[0], &undo);
    Move *best_rs; // how to use it?
    int score = -pv_split(search, board, depth - 1, ply + 1, -beta, -alpha, length - 1, best_rs);
    undo_move(board, &moves[0], &undo);
    if (score >= beta){
        return score;
    }
    if (score > alpha) {
        alpha = score;
        *best = moves[0];
    }
    
    treesplit(search, board, depth - 1, ply + 1, -beta, -alpha, result);
    //GPU_alpha_beta <<<1, count-1>>> (dev_search, dev_board, moves, depth -1, ply +1, -beta, -alpha, dev_scores);

    for (int i = 1; i < count; i++) {
        if (scores[i] > alpha) {
            alpha = scores[i];
            *best = moves[i];
        }
    }
    if (best) {
        memcpy(result, best, sizeof(Move));
    }

	return alpha;
}

int do_search(Search *search, Board *board) {
    int result = 1;
    double start = now();
    int score = 0;
    int depth = 5;
    int lo = INF;
    int hi = INF;
    int alpha = score - lo;
    int beta = score + hi;
    score = root_search(search, board, depth, 0, alpha, beta, &search->move);
    double elapsed = now() - start;
    if (search->uci) {
        char move_string[16];
        move_to_string(&search->move, move_string);
        int millis = elapsed * 1000;
        printf("info depth %d score cp %d time %d pv",
               depth, score, millis);
        printf("\n");
    }
    if (now() - start < 1) {
        sleep(1);
    }
    if (search->uci) {
        char move_string[16];
        move_to_string(&search->move, move_string);
        printf("bestmove %s\n", move_string);
    }
    return result;
}
