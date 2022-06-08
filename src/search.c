#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "search.h"
#include "eval.h"
#include "gen.h"
#include "move.h"
#include "util.h"

#define XOR_SWAP(a, b) a = a ^ b; b = a ^ b; a = a ^ b;

void sort_moves(Board *board, Move *moves, int count) {
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

int alpha_beta(Search *search, Board *board, int depth, int ply, int alpha, int beta) {
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

int root_search(Search *search, Board *board, int depth, int ply, int alpha, int beta, Move *result) {
    Undo undo;
    Move moves[MAX_MOVES];
    int count = gen_moves(board, moves);
    sort_moves(board, moves, count);
    Move *best = NULL;
    for (int i = 0; i < count; i++) {
        Move *move = &moves[i];
        do_move(board, move, &undo);
        int score = -alpha_beta(search, board, depth - 1, ply + 1, -beta, -alpha);
        undo_move(board, move, &undo);
        if (score > alpha) {
            alpha = score;
            best = move;
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
    int depth = 8;
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
