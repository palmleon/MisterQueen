#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "search.h"
#include "eval.h"
#include "gen.h"
#include "move.h"
#include "util.h"

#define LEN_POSITIONS 3
#define MAX_DEPTH 6

// RICORDA CHE IL PUNTEGGIO DELLA MOSSA VIENE VALUTATO NELLA SORT MOVES e integrato nella board_set()

//void alpha_beta_iter(Board *board_parent, int depth, int alpha_parent, int beta_parent, Move* moves_parent, int* scores_parent, int idx);


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
        result = evaluate(board); // + evaluate_pawns(board);
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
            //}
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

// Iterative Alpha-Beta Pruning

__device__ void alpha_beta_gpu_iter(Board *board_parent, int depth, int alpha_parent, int beta_parent, Move* moves_parent, int* scores_parent) {
    Board board = *board_parent;
    Move moves[MAX_MOVES * (MAX_DEPTH-1)];
    int can_move[MAX_DEPTH] = {0};
    Undo undo[MAX_DEPTH];
    int scores[MAX_DEPTH];
    int alpha[MAX_DEPTH];
    int beta[MAX_DEPTH];
    int beta_reached[MAX_DEPTH] = {0};
    int curr_depth = depth;
    int count = 0, board_illegal, curr_idx = -1;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //moves[0] = moves_parent[idx+1];
    alpha[curr_depth] = alpha_parent;
    beta[curr_depth] = beta_parent;
    do_move(&board, &(moves_parent[idx+1]), &undo[curr_depth]);
    // initial board created
    if(is_illegal(&board)){
        scores[curr_depth] = INF;
    }
    else if (curr_depth == 0) {
        scores[curr_depth] = evaluate(&board);
    }
    else {
        count = gen_moves(&board, &(moves[0]));
        curr_idx = count - 1;
    }

    while (curr_idx >= 0){ 
        count = -1;
        board_illegal = 0;
        while(count != 0 && !moves[curr_idx].already_executed) {
            count = 0;
            curr_depth--;
            do_move(&board, &(moves[curr_idx]), &undo[curr_depth]);
            can_move[curr_depth] = 0;
            beta_reached[curr_depth] = 0;
            alpha[curr_depth] = -beta[curr_depth+1];
            beta[curr_depth] = -alpha[curr_depth+1];
            board_illegal = is_illegal(&board);
            if (curr_depth > 0 && !board_illegal) {
                count = gen_moves(&board, &(moves[curr_idx+1]));
                curr_idx += count;
            }
        }
        //terminal nodes
        if (count == 0 && board_illegal) {
            scores[curr_depth] = INF;
        }
        else if (curr_depth == 0 || count == 0) { 
            scores[curr_depth] = evaluate(&board); 
        }

        if (curr_idx >= 0) {
            undo_move(&board, &(moves[curr_idx]), &(undo[curr_depth]));
            curr_depth++;
            curr_idx--;
        }
        // at this point, we should have the correct value of scores[curr_depth]
        if (-scores[curr_depth-1] > -INF) {
            can_move[curr_depth] = 1;
        }
        if (-scores[curr_depth-1] >= beta[curr_depth]) {
            scores[curr_depth] = beta[curr_depth];
            beta_reached[curr_depth] = 1;
            while (!moves[curr_idx].already_executed){
                curr_idx--;
            }
            //return beta;
            //final_score = beta;
            //beta_reached = 1;
            //break;
        }
        else if (-scores[curr_depth-1] > alpha[curr_depth]) {
            alpha[curr_depth] = -scores[curr_depth-1];
        }
        // all children executed, time to check... checks
        if (moves[curr_idx].already_executed || curr_idx < 0){  
            if(!beta_reached[curr_depth]){
                scores[curr_depth] = alpha[curr_depth];
                if(!can_move[curr_depth]){
                    if(is_check(&board, board.color)){
                        scores[curr_depth] = -MATE;
                    }
                    else {
                        scores[curr_depth] = 0;
                    }
                }
            }
        }
    }
    undo_move(&board, &(moves_parent[idx+1]), &undo[curr_depth]);
    scores_parent[idx+1] = -scores[curr_depth];
}

__global__ void alpha_beta_gpu_kernel(Board *board_parent, int depth, int alpha, int beta, Move* moves_parent, int* scores){
    alpha_beta_gpu_iter(board_parent, depth, alpha, beta, moves_parent, scores);
}

int alpha_beta_cpu(Board *board, int depth, int ply, int alpha, int beta, int *positions, int len_positions) {
    int result;
    //int num_multiproc;
    //cudaDeviceGetAttribute(&num_multiproc, cudaDevAttrMultiProcessorCount, 0);
    //printf("Multiprocessors: %d\n", num_multiproc);
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board); // + evaluate_pawns(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        //int scores[MAX_MOVES];
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
            alpha_beta_gpu_kernel<<<count-1, 1>>>(d_board, depth - 1, -beta, -alpha, d_moves, d_scores); // first move already counted
            //alpha_beta_gpu_kernel<<<num_multiproc, (count-1)/num_multiproc + 1>>>(d_board, depth - 1, ply + 1, -beta, -alpha, d_moves, d_scores, count-1);
            checkCudaErrors(cudaMemcpy(scores, d_scores, count * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_board); cudaFree(d_moves); cudaFree(d_scores);
            for (int i = 1; i < count; i++){
                //alpha_beta_gpu_iter(board, depth - 1, -beta, -alpha, moves, scores, i-1);
                //undo_move(board, move, &undo);
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
    //int num_multiproc;
    //cudaDeviceGetAttribute(&num_multiproc, cudaDevAttrMultiProcessorCount, 0);
    //printf("Multiprocessors: %d\n", num_multiproc);
    //int scores[MAX_MOVES];
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
        alpha_beta_gpu_kernel<<<count-1,1>>>(d_board, depth - 1, -beta, -alpha, d_moves, d_scores);
        //alpha_beta_gpu_kernel<<<num_multiproc, (count-1)/num_multiproc + 1>>>(d_board, depth - 1, ply + 1, -beta, -alpha, d_moves, d_scores, count-1);
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

int do_search(Search *search, Board *board) {
    struct timespec start, end;
    int result = 1;
    int score = 0;
    const int depth = MAX_DEPTH;
    int lo = INF;
    int hi = INF;
    int alpha = score - lo;
    int beta = score + hi;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    score = root_search(board, depth, 0, alpha, beta, &search->move);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    if (search->uci) {
        char move_string[16];
        move_to_string(&search->move, move_string);
        int millis = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
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

