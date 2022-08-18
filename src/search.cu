#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "search.h"
#include "config.h"
#include "eval.h"
#include "gen.h"
#include "gpu.h"
#include "move.h"
#include "util.h"

#define LEN_POSITIONS 3

void sort_moves(Board *board, Move *moves, int count) {
    int best = -INF, index;
    for (int i = 0; i < count; i++) {
        Move *move = &(moves[i]);
        int score = score_move(board, move);
        if (score > best) {
            best = score;
            index = i;
        }
    }
    if (count >= 1) {
        Move tmp;
        tmp = moves[index];
        moves[index] = moves[0];
        moves[0] = tmp;
    }
}

int initial_sort_moves_rec(Board *board, int *positions, int len, int ply, int alpha, int beta) {
    int result;
    if (is_illegal(board)) {
        result = INF;
    }
    else if (len <= 0) {
        result = evaluate(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int count = gen_moves(board, moves);
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
            if (is_check(board)) {
                result = -MATE + ply;
            } else {
                result = 0;
            }
        }
        free(best_indexes);
    }
    return result;
}

void initial_sort_moves(Board *board, Move *moves, int count, int *positions, int len) {
    Undo undo;
    int *best_indexes = (int*) malloc(sizeof(int)*(len-1));
    int best_score = -INF;
    for (int i = 0; i < count; i++) {
        Move *move = &(moves[i]);
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
    if (count >= 1){
        Move tmp;
        tmp = moves[positions[0]];
        moves[positions[0]] = moves[0];
        moves[0] = tmp;
    }
    free(best_indexes);
}

int alpha_beta_cpu(Board *board, int depth, int ply, int alpha, int beta, int *positions, int len_positions) {
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
        int count = gen_moves(board, moves);
        int *scores = (int*) malloc (count * sizeof(int));
        int can_move = 0;
        if (ply < len_positions) {
            Move tmp;
            tmp = moves[positions[ply]];
            moves[positions[ply]] = moves[0];
            moves[0] = tmp;
        }
        else {
            sort_moves(board, moves, count);
        }
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
            checkCudaErrors(cudaMalloc(&d_board, sizeof(Board)));
            checkCudaErrors(cudaMalloc(&d_moves, count * sizeof(Move)));
            checkCudaErrors(cudaMalloc(&d_scores, count * sizeof(int)));
            checkCudaErrors(cudaMemcpy(d_board, board, sizeof(Board), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_moves, moves, count * sizeof(Move), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_scores, scores, count * sizeof(int), cudaMemcpyHostToDevice));
            alpha_beta_gpu_kernel<<<count-1, dim3(1, THREADS_PER_NODE, 1), 64 * (sizeof(bb) + sizeof(int))>>>(d_board, depth - 1, -beta, -alpha, d_moves, d_scores); // first move already counted
            checkCudaErrors(cudaMemcpy(scores, d_scores, count * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_board); cudaFree(d_moves); cudaFree(d_scores);
            for (int i = 1; i < count; i++){
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

int root_search(Board *board, int depth, int ply, int alpha, int beta, Move *result) {
    Undo undo;
    Move moves[MAX_MOVES];
    int positions[LEN_POSITIONS];
    int count = gen_moves(board, moves);
    int *scores = (int*) malloc (count * sizeof(int));
    initial_sort_moves(board, moves, count, positions, LEN_POSITIONS);
    for (int i = 0; i < count; i++){
        moves[i].already_executed = 0; //search has not begun yet!
    }
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
        checkCudaErrors(cudaMalloc(&d_board, sizeof(Board)));
        checkCudaErrors(cudaMalloc(&d_moves, count * sizeof(Move)));
        checkCudaErrors(cudaMalloc(&d_scores, count * sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_board, board, sizeof(Board), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_moves, moves, count * sizeof(Move), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_scores, scores, count * sizeof(int), cudaMemcpyHostToDevice));
        alpha_beta_gpu_kernel<<<count-1, dim3(1, THREADS_PER_NODE, 1),  64 * (sizeof(bb)  + sizeof(int))>>>(d_board, depth - 1, -beta, -alpha, d_moves, d_scores);
        checkCudaErrors(cudaMemcpy(scores, d_scores, count * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_board); cudaFree(d_moves); cudaFree(d_scores);
        for (int i = 1; i < count; i++){
            if (scores[i] > alpha) {
                alpha = scores[i];
                best = &(moves[i]);
            }   
        }
    }
    if (best) {
        memcpy(result, best, sizeof(Move));
    }
    return alpha;
}

int do_search(Board *board, int uci, Move *move) {
    struct timespec start, end;
    int result = 1;
    int score = 0;
    const int depth = MAX_DEPTH;
    int lo = INF;
    int hi = INF;
    int alpha = score - lo;
    int beta = score + hi;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    score = root_search(board, depth, 0, alpha, beta, move);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    if (uci) {
        char move_string[16];
        move_to_string(move, move_string);
        int millis = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
        printf("Stats:\n| depth: %d\n| score: %d\n| time: %d ms\n",
               depth, score, millis);
        notate_move(board, move, move_string);
        printf("| best move: %s\n", move_string);
    }
    return result;

}