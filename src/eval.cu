#include "eval.h"

__device__ __host__ int count_stacked_pawns(bb pawns, int count) {
    int result = 0;
    result += BITS(pawns & FILE_A) == count;
    result += BITS(pawns & FILE_B) == count;
    result += BITS(pawns & FILE_C) == count;
    result += BITS(pawns & FILE_D) == count;
    result += BITS(pawns & FILE_E) == count;
    result += BITS(pawns & FILE_F) == count;
    result += BITS(pawns & FILE_G) == count;
    result += BITS(pawns & FILE_H) == count;
    return result;
}

__device__ __host__ int evaluate(Board *board) {
    int score = 0;
    // evaluate the total score square by square
    score += board->material;
    // evaluate position, square by square
    score += board->position;

    // evaluate stacked pawns
    score -= count_stacked_pawns(board->pawns & board->white, 2) * 50;
    score -= count_stacked_pawns(board->pawns & board->white, 3) * 100;
    score += count_stacked_pawns(board->pawns & board->black, 2) * 50;
    score += count_stacked_pawns(board->pawns & board->black, 3) * 100;
    return board->color ? -score : score;
}