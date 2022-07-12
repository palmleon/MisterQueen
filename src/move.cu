#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "board.h"
#include "gen.h"
#include "move.h"

__device__ __host__ void do_move(Board *board, Move *move, Undo *undo) {
    const bb rank_2nd[2] = {0x000000000000ff00L, 0x00ff000000000000L};
    const bb rank_4th[2] = {0x00000000ff000000L, 0x000000ff00000000L};
    const char castles_all[2] = {CASTLE_WHITE, CASTLE_BLACK};
    const char castles_all_types[4] = {CASTLE_WHITE_KING, CASTLE_WHITE_QUEEN, CASTLE_BLACK_KING, CASTLE_BLACK_QUEEN};
    const int king_start[2] = {4,60};
    const int king_arrival[4] = {6,2,62,58};
    const int rook_start[4] = {7,0,63,56};
    const int rook_arrival[4] = {5,3,61,59};
    const char rooks[2] = {WHITE_ROOK, BLACK_ROOK};
    const int color_bit = board->color / 8;
    const int coeff[2] = {1, -1};
    undo->piece = board_get_piece(board, move->src);
    undo->capture = board_get_piece(board, move->dst);
    undo->castle = board->castle;
    undo->ep = board->ep;
    move->already_executed = 1;
    Move move2 = *move;
    // remove the moving piece from its starting position
    board_set(board, move->src, EMPTY);
    // define the ending position of the piece
    if (move->promotion) {
        board_set(board, move->dst, move->promotion | board->color);
    }
    else {
        board_set(board, move->dst, undo->piece);
    }
    board->ep = 0L;
    // check for an en passant move
    if (PIECE(undo->piece) == PAWN) {
         bb src = BIT(move->src);
         bb dst = BIT(move->dst);
         if ((src & rank_2nd[color_bit]) && (dst & rank_4th[color_bit])){
            board->ep = BIT(move->src + coeff[color_bit]*8);
         }
         if (dst == undo->ep) {
            board_set(board, move->dst - coeff[color_bit]*8, EMPTY);
         }
    }
    // check for a castle
    else if (PIECE(undo->piece) == KING) {
        board->castle &= ~castles_all[color_bit];
        for (int i = 0; i < 2; i++) {
            if (move->src == king_start[color_bit] && move->dst == king_arrival[color_bit*2+i]){
                board_set(board, rook_start[2*color_bit+i], EMPTY);
                board_set(board, rook_arrival[2*color_bit+i], rooks[color_bit]);
            } 
        }
    }
    // consider the case where the rook moves or is eaten: the castle is no more possible
    for (int i = 0; i < 4; i++){
        if (move->src == rook_start[i] || move->dst == rook_start[i])
            board->castle &= ~castles_all_types[i];
    }

    board->color ^= BLACK;
}

__device__ __host__ void undo_move(Board *board, Move *move, Undo *undo) {
    //TOGGLE_HASH(board);
    const int coeff[2] = {1, -1};
    const int color_bit = (board->color / 8) ^ 1;
    const char pawns[2] = {WHITE_PAWN, BLACK_PAWN};
    const char rooks[2] = {WHITE_ROOK, BLACK_ROOK};
    const int king_start[2] = {4, 60};
    const int king_arrival[4] = {6,2,62,58};
    const int rook_start[4] = {7,0,63,56};
    const int rook_arrival[4] = {5,3,61,59};
    
    board_set(board, move->src, undo->piece);
    board_set(board, move->dst, undo->capture);
    board->castle = undo->castle;
    board->ep = undo->ep;
    if (PIECE(undo->piece) == PAWN){
        if (BIT(move->dst) == undo->ep){
            board_set(board, move->dst - coeff[color_bit]*8, pawns[color_bit^1]);
        }
    }
    else if (PIECE(undo->piece) == KING){
        for (int i = 0; i < 2; i++){
            if (move->src == king_start[color_bit] && move->dst == king_arrival[2*color_bit+i]){
                board_set(board, rook_start[2*color_bit+i], rooks[color_bit]);
                board_set(board, rook_arrival[2*color_bit+i], EMPTY);
            }
        }
    }
    board->color ^= BLACK;
}

// Compute score, but does not update it in the board struct
// Not important to optimize, as it is used only by the CPU
int score_move(Board *board, Move *move) {
    int result = 0;
    unsigned char src = move->src;
    unsigned char dst = move->dst;
    unsigned char piece = board_get_piece(board, src);
    unsigned char capture = board_get_piece(board, dst);
    int piece_material = 0;
    int capture_material = 0;
    //if (COLOR(piece)) {
        switch (PIECE(piece)) {
            case PAWN:
                piece_material = MATERIAL_PAWN;
                COLOR(piece) ?
                    (result += (POSITION_BLACK_PAWN[dst] - POSITION_BLACK_PAWN[src])) :
                    (result += (POSITION_WHITE_PAWN[dst] - POSITION_WHITE_PAWN[src]));
                //result -= POSITION_BLACK_PAWN[src];
                //result += POSITION_BLACK_PAWN[dst];
                break;
            case KNIGHT:
                piece_material = MATERIAL_KNIGHT;
                COLOR(piece) ?
                    (result += (POSITION_BLACK_KNIGHT[dst] - POSITION_BLACK_KNIGHT[src])) :
                    (result += (POSITION_WHITE_KNIGHT[dst] - POSITION_WHITE_KNIGHT[src]));
                //result -= POSITION_BLACK_KNIGHT[src];
                //result += POSITION_BLACK_KNIGHT[dst];
                break;
            case BISHOP:
                piece_material = MATERIAL_BISHOP;
                COLOR(piece) ?
                    (result += (POSITION_BLACK_BISHOP[dst] - POSITION_BLACK_BISHOP[src])) :
                    (result += (POSITION_WHITE_BISHOP[dst] - POSITION_WHITE_BISHOP[src]));
                //result -= POSITION_BLACK_BISHOP[src];
                //result += POSITION_BLACK_BISHOP[dst];
                break;
            case ROOK:
                piece_material = MATERIAL_ROOK;
                COLOR(piece) ?
                    (result += (POSITION_BLACK_ROOK[dst] - POSITION_BLACK_ROOK[src])) :
                    (result += (POSITION_WHITE_ROOK[dst] - POSITION_WHITE_ROOK[src]));
                //result -= POSITION_BLACK_ROOK[src];
                //result += POSITION_BLACK_ROOK[dst];
                break;
            case QUEEN:
                piece_material = MATERIAL_QUEEN;
                COLOR(piece) ?
                    (result += (POSITION_BLACK_QUEEN[dst] - POSITION_BLACK_QUEEN[src])) :
                    (result += (POSITION_WHITE_QUEEN[dst] - POSITION_WHITE_QUEEN[src]));
                //result -= POSITION_BLACK_QUEEN[src];
                //result += POSITION_BLACK_QUEEN[dst];
                break;
            case KING:
                piece_material = MATERIAL_KING;
                COLOR(piece) ?
                    (result += (POSITION_BLACK_KING[dst] - POSITION_BLACK_KING[src])) :
                    (result += (POSITION_WHITE_KING[dst] - POSITION_WHITE_KING[src]));
                //result -= POSITION_BLACK_KING[src];
                //result += POSITION_BLACK_KING[dst];
                break;
        }
    if (capture) {
             switch (PIECE(capture)) {
                case PAWN:
                    capture_material = MATERIAL_PAWN;
                    COLOR(capture) ? (result += POSITION_BLACK_PAWN[dst]) : (result += POSITION_WHITE_PAWN[dst]);
                    //result += POSITION_BLACK_PAWN[dst];
                    break;
                case KNIGHT:
                    capture_material = MATERIAL_KNIGHT;
                    COLOR(capture) ? (result += POSITION_BLACK_KNIGHT[dst]) : (result += POSITION_WHITE_KNIGHT[dst]);
                    //result += POSITION_BLACK_KNIGHT[dst];
                    break;
                case BISHOP:
                    capture_material = MATERIAL_BISHOP;
                    COLOR(capture) ? (result += POSITION_BLACK_BISHOP[dst]) : (result += POSITION_WHITE_BISHOP[dst]);
                    //result += POSITION_BLACK_BISHOP[dst];
                    break;
                case ROOK:
                    capture_material = MATERIAL_ROOK;
                    COLOR(capture) ? (result += POSITION_BLACK_ROOK[dst]) : (result += POSITION_WHITE_ROOK[dst]);
                    //result += POSITION_BLACK_ROOK[dst];
                    break;
                case QUEEN:
                    capture_material = MATERIAL_QUEEN;
                    COLOR(capture) ? (result += POSITION_BLACK_QUEEN[dst]) : (result += POSITION_WHITE_QUEEN[dst]);
                    //result += POSITION_BLACK_QUEEN[dst];
                    break;
                case KING:
                    capture_material = MATERIAL_KING;
                    COLOR(capture) ? (result += POSITION_BLACK_KING[dst]) : (result += POSITION_WHITE_KING[dst]);
                    //result += POSITION_BLACK_KING[dst];
                    break;
            }

        result += capture_material;
    }
    return result;
}

void move_to_string(Move *move, char *str) {
    char rank1 = '1' + move->src / 8;
    char file1 = 'a' + move->src % 8;
    char rank2 = '1' + move->dst / 8;
    char file2 = 'a' + move->dst % 8;
    *str++ = file1;
    *str++ = rank1;
    *str++ = file2;
    *str++ = rank2;
    if (move->promotion) {
        switch (move->promotion) {
            case KNIGHT: *str++ = 'n'; break;
            case BISHOP: *str++ = 'b'; break;
            case ROOK:   *str++ = 'r'; break;
            case QUEEN:  *str++ = 'q'; break;
        }
    }
    *str++ = 0;
}

void move_from_string(Move *move, const char *str) {
    int file1 = str[0] - 'a';
    int rank1 = str[1] - '1';
    int file2 = str[2] - 'a';
    int rank2 = str[3] - '1';
    int src = rank1 * 8 + file1;
    int dst = rank2 * 8 + file2;
    int promotion = EMPTY;
    switch (str[4]) {
        case 'n': promotion = KNIGHT; break;
        case 'b': promotion = BISHOP; break;
        case 'r': promotion = ROOK; break;
        case 'q': promotion = QUEEN; break;
    }
    move->src = src;
    move->dst = dst;
    move->promotion = promotion;
}


/*
 * Convert a move into standard notation
 */
void notate_move(Board *board, Move *move, char *result) {
    Move moves[MAX_MOVES];
    int count = gen_legal_moves(board, moves);
    char piece = board_get_piece(board, move->src);
    char capture = board_get_piece(board, move->dst);
    char rank1 = '1' + move->src / 8;
    char file1 = 'a' + move->src % 8;
    char rank2 = '1' + move->dst / 8;
    char file2 = 'a' + move->dst % 8;
    int show_rank1 = 0;
    int show_file1 = 0;
    if (PIECE(piece) == PAWN) {
        if (file1 != file2) {
            capture = 1;
        }
        if (capture) {
            show_file1 = 1;
        }
    }
    // ambiguity
    int ambiguous = 0;
    int unique_rank = 1;
    int unique_file = 1;
    for (int i = 0; i < count; i++) {
        Move *other = moves + i;
        if (move->dst != other->dst) {
            continue; // different target
        }
        if (move->src == other->src) {
            continue; // same move
        }
        if (piece != board_get_piece(board, other->src)) {
            continue; // different piece
        }
        ambiguous = 1;
        if (move->src % 8 == other->src % 8) {
            unique_file = 0;
        }
        if (move->src / 8 == other->src / 8) {
            unique_rank = 0;
        }
    }
    if (ambiguous) {
        if (unique_rank && unique_file) {
            show_file1 = 1;
        }
        else if (unique_rank) {
            show_rank1 = 1;
        }
        else if (unique_file) {
            show_file1 = 1;
        }
        else {
            show_rank1 = 1;
            show_file1 = 1;
        }
    }
    // castle
    int castle = 0;
    if (PIECE(piece) == KING) {
        castle = 1;
        if (move->src == 4 && move->dst == 6) {
            strcpy(result, "O-O");
            result += 3;
        }
        else if (move->src == 4 && move->dst == 2) {
            strcpy(result, "O-O-O");
            result += 5;
        }
        else if (move->src == 60 && move->dst == 62) {
            strcpy(result, "O-O");
            result += 3;
        }
        else if (move->src == 60 && move->dst == 58) {
            strcpy(result, "O-O-O");
            result += 5;
        }
        else {
            castle = 0;
        }
    }
    if (!castle) {
        // piece                                                                      
        switch (PIECE(piece)) {
            case KNIGHT: *result++ = 'N'; break;
            case BISHOP: *result++ = 'B'; break;
            case ROOK:   *result++ = 'R'; break;
            case QUEEN:  *result++ = 'Q'; break;
            case KING:   *result++ = 'K'; break;
        }
        // source
        if (show_file1) {
            *result++ = file1;
        }
        if (show_rank1) {
            *result++ = rank1;
        }
        // capture
        if (capture) {
            *result++ = 'x';
        }
        // target
        *result++ = file2;
        *result++ = rank2;
        // promotion
        if (move->promotion) {
            *result++ = '=';
            switch (move->promotion) {
                case KNIGHT: *result++ = 'N'; break;
                case BISHOP: *result++ = 'B'; break;
                case ROOK:   *result++ = 'R'; break;
                case QUEEN:  *result++ = 'Q'; break;
            }
        }
    }
    // check
    Undo undo;
    do_move(board, move, &undo);
    if (is_check(board, board->color)) {
        //if (has_legal_moves(board)) {
        if (gen_legal_moves(board, moves)){
            *result++ = '+';
        }
        else {
            *result++ = '#';
        }
    }
    undo_move(board, move, &undo);
    // null terminator
    *result++ = 0;
}

void print_move(Board *board, Move *move) {
    char notation[16];
    notate_move(board, move, notation);
    printf("%s", notation);
}

int parse_move(Board *board, const char *notation, Move *move) {
    char temp[16];
    Move moves[MAX_MOVES];
    int count = gen_legal_moves(board, moves);
    for (int i = 0; i < count; i++) {
        notate_move(board, &moves[i], temp);
        if (strcmp(notation, temp) == 0) {
            memcpy(move, &moves[i], sizeof(Move));
            return 1;
        }
    }
    return 0;
}
