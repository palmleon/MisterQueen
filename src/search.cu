#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "search.h"
#include "eval.h"
#include "gen.h"
#include "move.h"

#define LEN_POSITIONS 3

// RICORDA CHE IL PUNTEGGIO DELLA MOSSA VIENE VALUTATO NELLA SORT MOVES e integrato nella board_set()

void sort_moves(Board *board, Move *moves, int count) {
    int best = -INF, index;
    for (int i = 0; i < count; i++) {
        Move *move = moves + i;
        int score = score_move(board, move);
        if (score > best) {
            best = score;
            index = i;
        }
    }
    Move tmp;
    tmp = moves[index];
    moves[index] = moves[0];
    moves[0] = tmp;
}

int initial_sort_moves_rec(Board *board, int *positions, int len, int ply, int alpha, int beta) {
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (len <= 0) {
        result = evaluate(board); // + evaluate_pawns(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int count = gen_moves_new(board, moves);
        int *best_indexes = (int*) malloc(sizeof(int)*(len-1));
        int can_move = 0;
        for (int i = 0; i < count; i++) {
            Move *move = &moves[i];
            do_move(board, move, &undo);
            int score = -initial_sort_moves_rec(board, best_indexes, len - 1, ply + 1, -beta, -alpha);
            undo_move(board, move, &undo);
            if (score > -INF) {
                can_move = 1;
            }
            if (score >= beta) {
                return beta;
            }
            if (score > alpha) {
                alpha = score;
                positions[0] = i;
                for (int j = 1; j < len; j++) {
                    positions[j] = best_indexes[j-1];
                }
            }
        }
        result = alpha;
        if (!can_move) {
            //if (is_check(board)) {
            if (is_check(board, board->color)) {
                result = -MATE + ply;
            } else {
                result = 0;
            }
        }
    }
    return result;
}

void initial_sort_moves(Board *board, Move *moves, int count, int *positions, int len) {
    Undo undo;
    int *best_indexes = (int*) malloc(sizeof(int)*(len-1));
    int best_score = -INF;
    for (int i = 0; i < count; i++) {
        Move *move = moves + i;
        do_move(board, move, &undo);
        int score = -initial_sort_moves_rec(board, best_indexes, len-1, 1, -INF, +INF);
        undo_move(board, move, &undo);
        if (score > best_score) {
            best_score = score;
            positions[0] = i;
            for (int j = 1; j < len; j++) {
                positions[j] = best_indexes[j-1];
            }
        }
    }
    Move tmp;
    tmp = moves[positions[0]];
    moves[positions[0]] = moves[0];
    moves[0] = tmp;
    free(best_indexes);
}

__device__ int alpha_beta_gpu_device(Board *board, int depth, int ply, int alpha, int beta) {
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int count = gen_moves_new(board, moves);
        int can_move = 0;
        for (int i = 0; i < count; i++) {
            Move *move = &moves[i];
            do_move(board, move, &undo);
            int score = -alpha_beta_gpu_device(board, depth - 1, ply + 1, -beta, -alpha);
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
            //if (is_check(board)) {
            if (is_check(board, board->color)) {
                result = -MATE + ply;
            } else {
                result = 0;
            }
        }
    }
    return result;
}

__global__ void alpha_beta_gpu_kernel(Board *board_parent, int depth, int ply, int alpha, int beta, Move* moves_parent, int* scores, int nmoves) {
    //int result;
    Board board = *board_parent;
    Move moves[MAX_MOVES];
    Undo undo;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //for (int i = 0; i < nmoves; i++){
    do_move(&board, &(moves_parent[idx+1]), &undo);
    if (is_illegal(&board)) {
        scores[idx] = -INF;
    }
    else if (depth <= 0) {
        scores[idx] = -evaluate(&board);
    }
    else {
        Undo undo;
        int count = gen_moves_new(&board, moves);
        int can_move = 0;
        int beta_reached = 0;
        for (int i = 0; i < count && !beta_reached; i++) {
            Move *move = &(moves[i]);
            do_move(&board,  move, &undo);
            int score = -alpha_beta_gpu_device(&board, depth - 1, ply + 1, -beta, -alpha);
            undo_move(&board, move, &undo);
            if (score > -INF) {
                can_move = 1;
            }
            if (score >= beta) {
                //return beta;
                scores[idx] = -beta;
                beta_reached = 1;
                //break;
            }
            else if (score > alpha) {
                alpha = score;
            }
        }
        if (!beta_reached){
            scores[idx] = -alpha;
            //result = alpha;
            if (!can_move) {
                if (is_check(&board, board.color)) {
                    //result = -MATE + ply;
                    scores[idx] = MATE - ply;
                } else {
                    //result = 0;
                    scores[idx] = 0;
                }
            }
        }
    }
}

int alpha_beta_cpu(Board *board, int depth, int ply, int alpha, int beta, int *positions, int len_positions) {
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board); // + evaluate_pawns(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int scores[MAX_MOVES];
        int count = gen_moves_new(board, moves);
        int can_move = 0;
        if (ply < len_positions) {
            Move tmp;
            tmp = moves[positions[ply]];
            moves[positions[ply]] = moves[0];
            moves[0] = tmp;
        }
        else
            sort_moves(board, moves, count);
        
        if (count >= 1){
            do_move(board, &(moves[0]), &undo);
            int score = -alpha_beta_cpu(board, depth - 1, ply + 1, -beta, -alpha, positions, len_positions);
            undo_move(board, &(moves[0]), &undo);
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
        if (count > 1){ 
            Board *d_board;
            Move *d_moves;
            int *d_scores;
            cudaMalloc(&d_board, sizeof(Board));
            cudaMalloc(&d_moves, MAX_MOVES * sizeof(Move));
            cudaMalloc(&d_scores, MAX_MOVES * sizeof(int));
            cudaMemcpy(d_board, board, sizeof(Board), cudaMemcpyHostToDevice);
            cudaMemcpy(d_moves, moves, MAX_MOVES * sizeof(Move), cudaMemcpyHostToDevice);
            cudaMemcpy(d_scores, scores, MAX_MOVES * sizeof(int), cudaMemcpyHostToDevice);
            alpha_beta_gpu_kernel<<<count-1, 1>>>(d_board, depth - 1, ply + 1, -beta, -alpha, d_moves, d_scores, count-1); // first move already counted
            cudaMemcpy(scores, d_scores, MAX_MOVES * sizeof(int), cudaMemcpyDeviceToHost);
            //undo_move(board, move, &undo);
            for (int i = 0; i < count-1; i++){
                if (scores[i] > -INF) {
                    can_move = 1;
                }
                if (scores[i] >= beta) {
                    return beta;
                }
                if (scores[i] > alpha) {
                    alpha = scores[i];
                }
            }
            cudaFree(d_board); cudaFree(d_moves); cudaFree(d_scores);
        }
        result = alpha;
        if (!can_move) {
            //if (is_check(board)) {
            if (is_check(board, board->color)) {
                result = -MATE + ply;
            } else {
                result = 0;
            }
        }
    }
    return result;
}

int root_search(Board *board, int depth, int ply, int alpha, int beta, Move *result) {
    Undo undo;
    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int positions[LEN_POSITIONS];
    int count = gen_moves_new(board, moves);
    initial_sort_moves(board, moves, count, positions, LEN_POSITIONS);
    Move *best = NULL;
    if (count >= 1){
        do_move(board, &(moves[0]), &undo);
        int score = -alpha_beta_cpu(board, depth - 1, ply + 1, -beta, -alpha, positions, LEN_POSITIONS);
        undo_move(board, &(moves[0]), &undo);
        if (score > alpha) {
            alpha = score;
            best = &(moves[0]);
        }
    }
    if (count > 1){ // da cambiare con la chiamata al kernel
        Board *d_board;
        Move *d_moves;
        int *d_scores;
        cudaMalloc(&d_board, sizeof(Board));
        cudaMalloc(&d_moves, MAX_MOVES * sizeof(Move));
        cudaMalloc(&d_scores, MAX_MOVES * sizeof(int));
        cudaMemcpy(d_board, board, sizeof(Board), cudaMemcpyHostToDevice);
        cudaMemcpy(d_moves, moves, MAX_MOVES * sizeof(Move), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scores, scores, MAX_MOVES * sizeof(int), cudaMemcpyHostToDevice);
        alpha_beta_gpu_kernel<<<count-1,1>>>(d_board, depth - 1, ply + 1, -beta, -alpha, d_moves, d_scores, count-1);
        cudaMemcpy(scores, d_scores, MAX_MOVES * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < count - 1; i++){
            if (scores[i] > alpha) {
                alpha = scores[i];
                best = &(moves[i+1]);
            }   
        }
        cudaFree(d_board); cudaFree(d_moves); cudaFree(d_scores);
    }
    if (best) {
        memcpy(result, best, sizeof(Move));
    }
    return alpha;
}

int do_search(Search *search, Board *board) {
    int result = 1;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int score = 0;
    int depth = 6;
    int lo = INF;
    int hi = INF;
    int alpha = score - lo;
    int beta = score + hi;
    score = root_search(board, depth, 0, alpha, beta, &search->move);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);    
    u_int64_t elapsed = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    if (search->uci) {
        char move_string[16];
        move_to_string(&search->move, move_string);
        int millis = elapsed * 1000;
        printf("Stats:\n| depth: %d\n| score: %d\n| time: %d ms\n",
               depth, score, millis);
    }
    if (search->uci) {
        char move_string[16];
        notate_move(board, &search->move, move_string);
        //move_to_string(&search->move, move_string);
        printf("| best move: %s\n", move_string);
    }
    return result;
}
