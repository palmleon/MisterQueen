#include "eval.h"

int count_stacked_pawns(bb pawns, int count) {
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

int evaluate(Board *board) {
    int score = 0;
    // evaluate the total score square by square
    /*
    for (int sq = 0; sq < 64; sq++){
        char piece = board->squares[sq];
        // evaluate material
        int piece_value = piece_material[PIECE(piece)];
        int exp = COLOR(piece) >> 4;
        int coeff = pow(-1, exp);
        score += coeff * piece_value;
        // evaluate position
    }*/
    //score += board->white_material;
    //score -= board->black_material;
    score += board->material;
    // evaluate position, square by square
    //score += board->white_position;
    //score -= board->black_position;
    score += board->position;

    // evaluate stacked pawns
    score -= count_stacked_pawns(board->white_pawns, 2) * 50;
    score -= count_stacked_pawns(board->white_pawns, 3) * 100;
    score += count_stacked_pawns(board->black_pawns, 2) * 50;
    score += count_stacked_pawns(board->black_pawns, 3) * 100;
    //score += evaluate_pawns(board);
    return board->color ? -score : score;
}

/*
int evaluate_white_pawns(Board *board) {
    bb pawns = board->white_pawns;
    int score = 0;
    score -= count_stacked_pawns(pawns, 2) * 50;
    score -= count_stacked_pawns(pawns, 3) * 100;
    return score;
}

int evaluate_black_pawns(Board *board) {
    bb pawns = board->black_pawns;
    int score = 0;
    score -= count_stacked_pawns(pawns, 2) * 50;
    score -= count_stacked_pawns(pawns, 3) * 100;
    return score;
}

int evaluate_pawns(Board *board) {
    int score = 0;
    score += evaluate_white_pawns(board);
    score -= evaluate_black_pawns(board);
    return board->color ? -score : score;
}
*/