#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "search.h"
#include "eval.h"
#include "gen.h"
#include "move.h"
#include "util.h"

//#define XOR_SWAP(a, b) a = a ^ b; b = a ^ b; a = a ^ b;

#define LEN_POSITIONS 3

// RICORDA CHE IL PUNTEGGIO DELLA MOSSA VIENE VALUTATO NELLA SORT MOVES e integrato nella board_set()

/*void sort_moves(Board *board, Move *moves, int count) {
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
}*/

void sort_moves_new(Board *board, Move *moves, int count) {
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

/*
int alpha_beta(Search *search, Board *board, int depth, int ply, int alpha, int beta) {
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
*/

int alpha_beta_gpu_device(Board *board, int depth, int ply, int alpha, int beta) {
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

void alpha_beta_gpu_kernel(Board *board_parent, int depth, int ply, int alpha, int beta, Move* moves_parent, int* scores, int nmoves) {
    //int result;
    Board board = *board_parent;
    Move moves[MAX_MOVES];
    Undo undo;
    for (int i = 0; i < nmoves; i++){
        do_move(&board, &(moves_parent[i+1]), &undo);
        if (is_illegal(&board)) {
            scores[i] = -INF;
        }
        else if (depth <= 0) {
            scores[i] = -evaluate(&board);
        }
        else {
            Undo undo;
            int alpha_new = alpha;
            int count = gen_moves_new(&board, moves);
            int can_move = 0;
            int beta_reached = 0;
            for (int j = 0; j < count && !beta_reached; j++) {
                Move *move = &(moves[j]);
                do_move(&board, move, &undo);
                int score = -alpha_beta_gpu_device(&board, depth - 1, ply + 1, -beta, -alpha_new);
                undo_move(&board, move, &undo);
                if (score > -INF) {
                    can_move = 1;
                }
                if (score >= beta) {
                    //return beta;
                    scores[i] = -beta;
                    beta_reached = 1;
                    //break;
                }
                else if (score > alpha_new) {
                    alpha_new = score;
                    //scores[i] = -alpha;
                }
            }
            if (!beta_reached){
                scores[i] = -alpha_new;
                //result = alpha;
                if (!can_move) {
                    //if (is_check(board)) {
                    if (is_check(&board, board.color)) {
                        //result = -MATE + ply;
                        scores[i] = MATE - ply;
                    } else {
                        //result = 0;
                        scores[i] = 0;
                    }
                }
            }
        }
        undo_move(&board, &(moves_parent[i+1]), &undo);
    }
    //return result;
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
            sort_moves_new(board, moves, count);
        //for (int i = 0; i < count; i++) {
        //    Move *move = &moves[i];
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
        if (count > 1){ // da cambiare con la chiamata al kernel e conseguente gestione delle mosse
            //do_move(board, move, &undo);
            //int score = -alpha_beta_gpu_kernel(board, depth - 1, ply + 1, -beta, -alpha, moves, scores, count);
            alpha_beta_gpu_kernel(board, depth - 1, ply + 1, -beta, -alpha, moves, scores, count-1); // first move already counted
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
    //for (int i = 0; i < count; i++) {
    //    Move *move = &moves[i];
    //if (i == 0){
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
        //do_move(board, move, &undo);
        //int score = -alpha_beta_gpu_kernel(board, depth - 1, ply + 1, -beta, -alpha);
        alpha_beta_gpu_kernel(board, depth - 1, ply + 1, -beta, -alpha, moves, scores, count-1);
        //undo_move(board, move, &undo);
        for (int i = 0; i < count - 1; i++){
            if (scores[i] > alpha) {
                alpha = scores[i];
                best = &(moves[i+1]);
            }   
        }
    }
    //}
    if (best) {
        memcpy(result, best, sizeof(Move));
    }
    return alpha;
}

int do_search(Search *search, Board *board) {
    int result = 1;
    double start = now();
    int score = 0;
    int depth = 6;
    int lo = INF;
    int hi = INF;
    int alpha = score - lo;
    int beta = score + hi;
    score = root_search(board, depth, 0, alpha, beta, &search->move);
    double elapsed = now() - start;
    if (search->uci) {
        char move_string[16];
        move_to_string(&search->move, move_string);
        int millis = elapsed * 1000;
        printf("Stats:\n| depth: %d\n| score: %d\n| time: %d ms\n",
               depth, score, millis);
    }
    if (now() - start < 1) {
        sleep(1);
    }
    if (search->uci) {
        char move_string[16];
        notate_move(board, &search->move, move_string);
        //move_to_string(&search->move, move_string);
        printf("| best move: %s\n", move_string);
    }
    return result;
}
