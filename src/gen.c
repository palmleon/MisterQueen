#include <string.h>
#include <math.h>
#include "gen.h"


#define EMIT_MOVE(m, a, b) \
    (m)->src = (a); \
    (m)->dst = (b); \
    (m)->promotion = EMPTY; \
    (m)++;

#define EMIT_PROMOTION(m, a, b, p) \
    (m)->src = (a); \
    (m)->dst = (b); \
    (m)->promotion = (p); \
    (m)++;

#define EMIT_PROMOTIONS(m, a, b) \
    EMIT_PROMOTION(m, a, b, QUEEN) \
    EMIT_PROMOTION(m, a, b, ROOK) \
    EMIT_PROMOTION(m, a, b, BISHOP) \
    EMIT_PROMOTION(m, a, b, KNIGHT)

/*
// generic move generators
int gen_knight_moves(Move *moves, bb srcs, bb mask) {
    Move *ptr = moves;
    int src, dst;
    while (srcs) {
        POP_LSB(src, srcs);
        bb dsts = BB_KNIGHT[src] & mask;
        while (dsts) {
            POP_LSB(dst, dsts);
            EMIT_MOVE(moves, src, dst);
        }
    }
    return moves - ptr;
}

int gen_bishop_moves(Move *moves, bb srcs, bb mask, bb all) {
    Move *ptr = moves;
    int src, dst;
    while (srcs) {
        POP_LSB(src, srcs);
        bb dsts = bb_bishop(src, all) & mask;
        while (dsts) {
            POP_LSB(dst, dsts);
            EMIT_MOVE(moves, src, dst);
        }
    }
    return moves - ptr;
}

int gen_rook_moves(Move *moves, bb srcs, bb mask, bb all) {
    Move *ptr = moves;
    int src, dst;
    while (srcs) {
        POP_LSB(src, srcs);
        bb dsts = bb_rook(src, all) & mask;
        while (dsts) {
            POP_LSB(dst, dsts);
            EMIT_MOVE(moves, src, dst);
        }
    }
    return moves - ptr;
}

int gen_queen_moves(Move *moves, bb srcs, bb mask, bb all) {
    Move *ptr = moves;
    int src, dst;
    while (srcs) {
        POP_LSB(src, srcs);
        bb dsts = bb_queen(src, all) & mask;
        while (dsts) {
            POP_LSB(dst, dsts);
            EMIT_MOVE(moves, src, dst);
        }
    }
    return moves - ptr;
}

int gen_king_moves(Move *moves, bb srcs, bb mask) {
    Move *ptr = moves;
    int src, dst;
    while (srcs) {
        POP_LSB(src, srcs);
        bb dsts = BB_KING[src] & mask;
        while (dsts) {
            POP_LSB(dst, dsts);
            EMIT_MOVE(moves, src, dst);
        }
    }
    return moves - ptr;
}

// white move generators
int gen_white_pawn_moves(Board *board, Move *moves) {
    Move *ptr = moves;
    //bb pawns = board->white_pawns;
    bb pawns = board->pawns & board->white;
    bb mask = board->black | board->ep;
    bb promo = 0xff00000000000000L;
    bb p1 = (pawns << 8) & ~board->all; // pedoni avanti di una casella
    bb p2 = ((p1 & 0x0000000000ff0000L) << 8) & ~board->all; // pedoni in seconda riga
    bb a1 = ((pawns & 0xfefefefefefefefeL) << 7) & mask; // attacchi a dx
    bb a2 = ((pawns & 0x7f7f7f7f7f7f7f7fL) << 9) & mask; // attacchi a sx
    int sq;
    while (p1) {
        POP_LSB(sq, p1);
        if (BIT(sq) & promo) {
            EMIT_PROMOTIONS(moves, sq - 8, sq);
        }
        else {
            EMIT_MOVE(moves, sq - 8, sq);
        }
    }
    while (p2) {
        POP_LSB(sq, p2);
        EMIT_MOVE(moves, sq - 16, sq);
    }
    while (a1) {
        POP_LSB(sq, a1);
        if (BIT(sq) & promo) {
            EMIT_PROMOTIONS(moves, sq - 7, sq);
        }
        else {
            EMIT_MOVE(moves, sq - 7, sq);
        }
    }
    while (a2) {
        POP_LSB(sq, a2);
        if (BIT(sq) & promo) {
            EMIT_PROMOTIONS(moves, sq - 9, sq);
        }
        else {
            EMIT_MOVE(moves, sq - 9, sq);
        }
    }
    return moves - ptr;
}

int gen_white_knight_moves(Board *board, Move *moves) {
    return gen_knight_moves(
      //  moves, board->white_knights, ~board->white);
          moves, board->white & board->knights, ~board->white);
}

int gen_white_bishop_moves(Board *board, Move *moves) {
    return gen_bishop_moves(
     //   moves, board->white_bishops, ~board->white, board->all);
          moves, board->white & board->bishops, ~board->white, board->all);
}

int gen_white_rook_moves(Board *board, Move *moves) {
    return gen_rook_moves(
     //   moves, board->white_rooks, ~board->white, board->all);
          moves, board->white & board->rooks, ~board->white, board->all);
}

int gen_white_queen_moves(Board *board, Move *moves) {
    return gen_queen_moves(
     //   moves, board->white_queens, ~board->white, board->all);
          moves, board->white & board->queens, ~board->white, board->all);
}

int gen_white_king_moves(Board *board, Move *moves) {
    return gen_king_moves(
     //   moves, board->white_kings, ~board->white);
          moves, board->white & board->kings, ~board->white);
}

int gen_white_king_castles(Board *board, Move *moves) {
    Move *ptr = moves;
    if (board->castle & CASTLE_WHITE_KING) {
        if (!(board->all & 0x0000000000000060L)) {
            Move dummy[MAX_MOVES];
            //bb mask = 0x0000000000000030L;
            bb mask = 0x0000000000000070L;
            if (!gen_black_attacks_against(board, dummy, mask)) {
                EMIT_MOVE(moves, 4, 6);
            }
        }
    }
    if (board->castle & CASTLE_WHITE_QUEEN) {
        if (!(board->all & 0x000000000000000eL)) {
            Move dummy[MAX_MOVES];
            //bb mask = 0x0000000000000018L;
            bb mask = 0x000000000000001cL;
            if (!gen_black_attacks_against(board, dummy, mask)) {
                EMIT_MOVE(moves, 4, 2);
            }
        }
    }
    return moves - ptr;
}

int gen_white_moves(Board *board, Move *moves) {
    Move *ptr = moves;
    moves += gen_white_pawn_moves(board, moves);
    moves += gen_white_knight_moves(board, moves);
    moves += gen_white_bishop_moves(board, moves);
    moves += gen_white_rook_moves(board, moves);
    moves += gen_white_queen_moves(board, moves);
    moves += gen_white_king_moves(board, moves);
    moves += gen_white_king_castles(board, moves);
    return moves - ptr;
}

// white attack generators
int gen_white_pawn_attacks_against(Board *board, Move *moves, bb mask) {
    Move *ptr = moves;
    // bb pawns = board->white_pawns;
    bb pawns = board->pawns & board->white;
    bb a1 = ((pawns & 0xfefefefefefefefeL) << 7) & mask;
    bb a2 = ((pawns & 0x7f7f7f7f7f7f7f7fL) << 9) & mask;
    int sq;
    while (a1) {
        POP_LSB(sq, a1);
        EMIT_MOVE(moves, sq - 7, sq);
    }
    while (a2) {
        POP_LSB(sq, a2);
        EMIT_MOVE(moves, sq - 9, sq);
    }
    return moves - ptr;
}

int gen_white_knight_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_knight_moves(
     //   moves, board->white_knights, mask);
         moves, board->white & board->knights, mask);
}

int gen_white_bishop_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_bishop_moves(
     //   moves, board->white_bishops, mask, board->all);
          moves, board->white & board->bishops, mask, board->all);
}

int gen_white_rook_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_rook_moves(
     //   moves, board->white_rooks, mask, board->all);
          moves, board->white & board->rooks, mask, board->all);
}

int gen_white_queen_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_queen_moves(
      //  moves, board->white_queens, mask, board->all);
          moves, board->white & board->queens, mask, board->all);
}

int gen_white_king_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_king_moves(
     //   moves, board->white_kings, mask);
          moves, board->kings & board->white, mask);
}

int gen_white_attacks_against(Board *board, Move *moves, bb mask) {
    Move *ptr = moves;
    moves += gen_white_pawn_attacks_against(board, moves, mask);
    moves += gen_white_knight_attacks_against(board, moves, mask);
    moves += gen_white_bishop_attacks_against(board, moves, mask);
    moves += gen_white_rook_attacks_against(board, moves, mask);
    moves += gen_white_queen_attacks_against(board, moves, mask);
    moves += gen_white_king_attacks_against(board, moves, mask);
    return moves - ptr;
}

int gen_white_attacks(Board *board, Move *moves) {
    return gen_white_attacks_against(board, moves, board->black);
}

int gen_white_checks(Board *board, Move *moves) {
    // return gen_white_attacks_against(board, moves, board->black_kings);
    return gen_white_attacks_against(board, moves, board->black & board->kings);

}

// black move generators
int gen_black_pawn_moves(Board *board, Move *moves) {
    Move *ptr = moves;
    //bb pawns = board->black_pawns;
    bb pawns = board->black & board->pawns;
    bb mask = board->white | board->ep;
    bb promo = 0x00000000000000ffL;
    bb p1 = (pawns >> 8) & ~board->all;
    bb p2 = ((p1 & 0x0000ff0000000000L) >> 8) & ~board->all;
    bb a1 = ((pawns & 0x7f7f7f7f7f7f7f7fL) >> 7) & mask;
    bb a2 = ((pawns & 0xfefefefefefefefeL) >> 9) & mask;
    int sq;
    while (p1) {
        POP_LSB(sq, p1);
        if (BIT(sq) & promo) {
            EMIT_PROMOTIONS(moves, sq + 8, sq);
        }
        else {
            EMIT_MOVE(moves, sq + 8, sq);
        }
    }
    while (p2) {
        POP_LSB(sq, p2);
        EMIT_MOVE(moves, sq + 16, sq);
    }
    while (a1) {
        POP_LSB(sq, a1);
        if (BIT(sq) & promo) {
            EMIT_PROMOTIONS(moves, sq + 7, sq);
        }
        else {
            EMIT_MOVE(moves, sq + 7, sq);
        }
    }
    while (a2) {
        POP_LSB(sq, a2);
        if (BIT(sq) & promo) {
            EMIT_PROMOTIONS(moves, sq + 9, sq);
        }
        else {
            EMIT_MOVE(moves, sq + 9, sq);
        }
    }
    return moves - ptr;
}

int gen_black_knight_moves(Board *board, Move *moves) {
    return gen_knight_moves(
     //   moves, board->black_knights, ~board->black);
        moves, board->black & board->knights, ~board->black);
}

int gen_black_bishop_moves(Board *board, Move *moves) {
    return gen_bishop_moves(
      //  moves, board->black_bishops, ~board->black, board->all);
        moves, board->black & board->bishops, ~board->black, board->all);
}

int gen_black_rook_moves(Board *board, Move *moves) {
    return gen_rook_moves(
      //  moves, board->black_rooks, ~board->black, board->all);
        moves, board->black & board->rooks, ~board->black, board->all);
}

int gen_black_queen_moves(Board *board, Move *moves) {
    return gen_queen_moves(
      //  moves, board->black_queens, ~board->black, board->all);
        moves, board->black & board->queens, ~board->black, board->all);
}

int gen_black_king_moves(Board *board, Move *moves) {
    return gen_king_moves(
      //  moves, board->black_kings, ~board->black);
        moves, board->black & board->kings, ~board->black);
}

int gen_black_king_castles(Board *board, Move *moves) {
    Move *ptr = moves;
    if (board->castle & CASTLE_BLACK_KING) {
        if (!(board->all & 0x6000000000000000L)) {
            Move dummy[MAX_MOVES];
            //bb mask = 0x3000000000000000L;
            bb mask = 0x7000000000000000L;
            if (!gen_white_attacks_against(board, dummy, mask)) {
                EMIT_MOVE(moves, 60, 62);
            }
        }
    }
    if (board->castle & CASTLE_BLACK_QUEEN) {
        if (!(board->all & 0x0e00000000000000L)) {
            Move dummy[MAX_MOVES];
            //bb mask = 0x1800000000000000L;
            bb mask = 0x1c00000000000000L;
            if (!gen_white_attacks_against(board, dummy, mask)) {
                EMIT_MOVE(moves, 60, 58);
            }
        }
    }
    return moves - ptr;
}

int gen_black_moves(Board *board, Move *moves) {
    Move *ptr = moves;
    moves += gen_black_pawn_moves(board, moves);
    moves += gen_black_knight_moves(board, moves);
    moves += gen_black_bishop_moves(board, moves);
    moves += gen_black_rook_moves(board, moves);
    moves += gen_black_queen_moves(board, moves);
    moves += gen_black_king_moves(board, moves);
    moves += gen_black_king_castles(board, moves);
    return moves - ptr;
}

// black attack generators
int gen_black_pawn_attacks_against(Board *board, Move *moves, bb mask) {
    Move *ptr = moves;
   // bb pawns = board->black_pawns;
   bb pawns = board->pawns & board->black;
    bb a1 = ((pawns & 0x7f7f7f7f7f7f7f7fL) >> 7) & mask;
    bb a2 = ((pawns & 0xfefefefefefefefeL) >> 9) & mask;
    int sq;
    while (a1) {
        POP_LSB(sq, a1);
        EMIT_MOVE(moves, sq + 7, sq);
    }
    while (a2) {
        POP_LSB(sq, a2);
        EMIT_MOVE(moves, sq + 9, sq);
    }
    return moves - ptr;
}

int gen_black_knight_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_knight_moves(
     //   moves, board->black_knights, mask);
          moves, board->black & board->knights, mask);
}

int gen_black_bishop_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_bishop_moves(
      //  moves, board->black_bishops, mask, board->all);
          moves, board->black & board->bishops, mask, board->all);
}

int gen_black_rook_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_rook_moves(
      //  moves, board->black_rooks, mask, board->all);
          moves, board->black & board->rooks, mask, board->all);
}

int gen_black_queen_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_queen_moves(
      //  moves, board->black_queens, mask, board->all);
          moves, board->black & board->queens, mask, board->all);
}

int gen_black_king_attacks_against(Board *board, Move *moves, bb mask) {
    return gen_king_moves(
     //   moves, board->black_kings, mask);
        moves, board->black & board->kings, mask);
}

int gen_black_attacks_against(Board *board, Move *moves, bb mask) {
    Move *ptr = moves;
    moves += gen_black_pawn_attacks_against(board, moves, mask);
    moves += gen_black_knight_attacks_against(board, moves, mask);
    moves += gen_black_bishop_attacks_against(board, moves, mask);
    moves += gen_black_rook_attacks_against(board, moves, mask);
    moves += gen_black_queen_attacks_against(board, moves, mask);
    moves += gen_black_king_attacks_against(board, moves, mask);
    return moves - ptr;
}

int gen_black_attacks(Board *board, Move *moves) {
    return gen_black_attacks_against(board, moves, board->white);
}

int gen_black_checks(Board *board, Move *moves) {
    //return gen_black_attacks_against(board, moves, board->white_kings);
    return gen_black_attacks_against(board, moves, board->white & board->kings);

}

// color determined by board
int gen_moves(Board *board, Move *moves) {
    if (board->color) {
        return gen_black_moves(board, moves);
    }
    else {
        return gen_white_moves(board, moves);
    }
}

int gen_attacks(Board *board, Move *moves) {
    if (board->color) {
        return gen_black_attacks(board, moves);
    }
    else {
        return gen_white_attacks(board, moves);
    }
}

int gen_checks(Board *board, Move *moves) {
    if (board->color) {
        return gen_black_checks(board, moves);
    }
    else {
        return gen_white_checks(board, moves);
    }
}
*/

/*
int is_check(Board *board) {
    Move moves[MAX_MOVES];
    if (board->color) {
        return gen_white_checks(board, moves);
    }
    else {
        return gen_black_checks(board, moves);
    }
}

int is_illegal(Board *board) {
    Move moves[MAX_MOVES];
    if (board->color) {
        return gen_black_checks(board, moves);
    }
    else {
        return gen_white_checks(board, moves);
    }
}

int has_legal_moves(Board *board) {
    Move moves[MAX_MOVES];
    return gen_legal_moves(board, moves);
}
*/

/* To verify if the current player is in check, 
   we generate all the opponent moves 
   and verify if any of them can directly attack the king
   color: player that performs the check
*/
int is_check(Board *board, char color){
    // for black, board->color >> 4 = 0x01
    // for white, board->color >> 4 = 0x00
    const int color_bit = color >> 4;
    // coeff = -1 for white, +1 for black
    //const int coeff[2] = {-1, 1};
    const bb players_pieces[2] = {board->white, board->black}; // array defined to avoid an if-else
    const bb promo[2] = {0xff00000000000000L, 0x00000000000000ffL}; // representation of the promotion rank
    const bb third_rank[2] = {0x0000000000ff0000L, 0x0000ff0000000000L}; // used for initial double move of pawn
    const bb front_right_mask[2] = {0xfefefefefefefefeL, 0x7f7f7f7f7f7f7f7fL};
    const bb front_left_mask[2] = {0x7f7f7f7f7f7f7f7fL, 0xfefefefefefefefeL};
    const bb own_pieces = players_pieces[color_bit];
    const bb opponent_pieces = players_pieces[color_bit ^ 1];
    bb mask = ~own_pieces;
    const bb mask_pawn = opponent_pieces | board->ep;
    const bb mask_pawn_opp = own_pieces | board->ep;
    bb dsts = 0;

    for(int sq = 0; sq < 64; sq++){
        char piece = board->squares[sq];
        if (COLOR(piece) == color){
            bb pawn_bb;
            switch(PIECE(piece)){
                case PAWN:
                    pawn_bb = BIT(sq);
                    bb p1_vec[2] = {pawn_bb << 8, pawn_bb >> 8};
                    bb p1 = p1_vec[color_bit] & ~board->all;
                    bb p2 = p1 & third_rank[color_bit];
                    bb p2_vec[2] = {p2 << 8, p2 >> 8};
                    p2 = p2_vec[color_bit] & ~board->all;
                    bb a1 = pawn_bb & front_right_mask[color_bit];
                    bb a1_vec[2] = {a1 << 7, a1 >> 7};
                    a1 = a1_vec[color_bit] & mask_pawn;
                    bb a2 = pawn_bb & front_left_mask[color_bit];
                    bb a2_vec[2] = {a2 << 9, a2 >> 9};
                    a2 = a2_vec[color_bit] & mask_pawn;
                    dsts |= p1;
                    dsts |= p2;
                    dsts |= a1;
                    dsts |= a2;
                    break;                    
                case KNIGHT:
                    dsts |= BB_KNIGHT[sq] & mask;
                    break;
                case BISHOP:
                    dsts |= bb_bishop(sq, board->all) & mask;
                    break;
                case ROOK:
                    dsts |= bb_rook(sq, board->all) & mask;
                    break;
                case QUEEN:
                    dsts |= bb_queen(sq, board->all) & mask;
                    break;
                case KING:
                    dsts |= BB_KING[sq] & mask;
                    break;
                default: // empty piece
                    break;
            }
        }
    }
    return (dsts & opponent_pieces & board->kings) != (long long) 0;
}

int is_illegal(Board *board){
    return is_check(board, board->color);
}

/*
 * Move generation algorithm
 * To make it as parallelizable as possible, it has been conceived as an iteration
 * on all the squares of the board. If there is a piece on the board of the same color
 * as of the current player (i.e. who must move), then I generate all the possible moves
 * for that piece.
 * Since the Board representation is based on bitboards, it is possible to compute
 * all the possible moves using tables, built at initialization time.
 * Each of these tables is a bitmap, representing where it is possible to move
 * To better understand the tables, read the bb_init() function in bb.c
 */
int gen_moves_new(Board *board, Move *moves){
    Move *ptr = moves;
    // for black, board->color >> 4 = 0x01
    // for white, board->color >> 4 = 0x00
    const int color_bit = board->color >> 4;
    // coeff = -1 for white, +1 for black
    //const int coeff[2] = {-1, 1};
    const bb players_pieces[2] = {board->white, board->black}; // array defined to avoid an if-else
    const bb promo[2] = {0xff00000000000000L, 0x00000000000000ffL}; // representation of the promotion rank
    const bb third_rank[2] = {0x0000000000ff0000L, 0x0000ff0000000000L}; // used for initial double move of pawn
    const bb front_right_mask[2] = {0xfefefefefefefefeL, 0x7f7f7f7f7f7f7f7fL};
    const bb front_left_mask[2] = {0x7f7f7f7f7f7f7f7fL, 0xfefefefefefefefeL};
    const bb own_pieces = players_pieces[color_bit];
    const bb opponent_pieces = players_pieces[color_bit ^ 1];
    bb mask = ~own_pieces;
    const bb mask_pawn = opponent_pieces | board->ep;
    const bb mask_pawn_opp = own_pieces | board->ep;
    const char castles[4] = {CASTLE_WHITE_KING, CASTLE_WHITE_QUEEN, CASTLE_BLACK_KING, CASTLE_BLACK_QUEEN};
    const bb castle_masks_1[4] = {0x0000000000000060L, 0x000000000000000eL, 0x6000000000000000L, 0x0e00000000000000L};
    const bb castle_masks_2[4] = {0x0000000000000070L, 0x000000000000001cL, 0x7000000000000000L, 0x1c00000000000000L};
    const int castle_king_pos_before[2] = {4, 60};
    const int castle_king_pos_after[4] = {6, 2, 62, 58};

    for(int sq = 0; sq < 64; sq++){
        char piece = board->squares[sq];
        bb dsts = 0;
        // move a piece only if it is of the current moving player!
        if (COLOR(piece) == board->color){
            bb pawn_bb;
            switch(PIECE(piece)){
                case PAWN:
                    /*
                    bb p1 = (pawn_bb >> (coeff[color_bit]*8)) & ~board->all;
                    bb p2 = ((p1 & third_rank[color_bit]) >> (coeff[color_bit]*8)) & ~board->all;
                    bb a1 = ((pawn_bb & front_right_mask[color_bit]) >> (coeff[color_bit]*7)) & mask_pawn;
                    bb a2 = ((pawn_bb & front_left_mask[color_bit]) >> (coeff[color_bit]*9)) & mask_pawn;
                    */
                    pawn_bb = BIT(sq);
                    bb p1_vec[2] = {pawn_bb << 8, pawn_bb >> 8};
                    bb p1 = p1_vec[color_bit] & ~board->all;
                    bb p2 = p1 & third_rank[color_bit];
                    bb p2_vec[2] = {p2 << 8, p2 >> 8};
                    p2 = p2_vec[color_bit] & ~board->all;
                    bb a1 = pawn_bb & front_right_mask[color_bit];
                    bb a1_vec[2] = {a1 << 7, a1 >> 7};
                    a1 = a1_vec[color_bit] & mask_pawn;
                    bb a2 = pawn_bb & front_left_mask[color_bit];
                    bb a2_vec[2] = {a2 << 9, a2 >> 9};
                    a2 = a2_vec[color_bit] & mask_pawn;
                    dsts |= p1;
                    dsts |= p2;
                    dsts |= a1;
                    dsts |= a2;
                    break;
                    
                case KNIGHT:
                    dsts = BB_KNIGHT[sq] & mask;
                    break;
                case BISHOP:
                    dsts = bb_bishop(sq, board->all) & mask;
                    break;
                case ROOK:
                    dsts = bb_rook(sq, board->all) & mask;
                    break;
                case QUEEN:
                    dsts = bb_queen(sq, board->all) & mask;
                    break;
                case KING:
                    dsts = BB_KING[sq] & mask;
                    break;
                default: // empty piece
                    break;
            }
            // Emit all the moves
            while (dsts) {
                int dst;
                POP_LSB(dst, dsts);
                if ((PIECE(piece) == PAWN) && (BIT(dst) & promo[color_bit])){
                    EMIT_PROMOTIONS(moves, sq, dst);
                }
                else {
                    EMIT_MOVE(moves, sq, dst);
                }
            }
        }
    }
    // GENERATE CASTLES
    mask = ~opponent_pieces;
    // look for opponent attacks in the squares where the king should move
    bb dsts = 0;
    if ((board->castle & castles[color_bit*2]) || board->castle & castles[color_bit*2+1]){
        for (int sq = 0; sq < 64; sq++){
            char piece = board->squares[sq];
            if (COLOR(piece) != board->color){
                bb pawn_bb;
                switch(PIECE(piece)){
                    case PAWN:
                        pawn_bb = BIT(sq);
                        /*
                        bb a1 = ((pawn_bb & front_right_mask[color_bit^1]) >> coeff[color_bit^1]*7) & mask_pawn_opp;
                        bb a2 = ((pawn_bb & front_left_mask[color_bit^1]) >> coeff[color_bit^1]*9) & mask_pawn_opp;
                        */
                        bb a1 = pawn_bb & front_right_mask[color_bit ^ 1];
                        bb a1_vec[2] = {a1 << 7, a1 >> 7};
                        a1 = a1_vec[color_bit^1] & mask_pawn_opp;
                        bb a2 = pawn_bb & front_left_mask[color_bit ^ 1];
                        bb a2_vec[2] = {a2 << 9, a2 >> 9};
                        a2 = a2_vec[color_bit^1] & mask_pawn_opp;
                        dsts |= a1 | a2;
                        //bb pawn_one_step_forward = pawn_bb
                        break;
                    case KNIGHT:
                        dsts |= BB_KNIGHT[sq] & mask;
                        break;
                    case BISHOP:
                        dsts |= bb_bishop(sq, board->all) & mask;
                        break;
                    case ROOK:
                        dsts |= bb_rook(sq, board->all) & mask;
                        break;
                    case QUEEN:
                        dsts |= bb_queen(sq, board->all) & mask;
                        break;
                    case KING:
                        dsts |= BB_KING[sq] & mask;
                        break;
                    default: // empty piece
                        break;
                }
            }
        }
        for (int i = 0; i < 2; i++) {
            // check if the player can castle and, if that is the case,
            // where it can castle and whether there are pieces
            // between the king and the rook
            bb mask = castle_masks_2[color_bit*2+i];
            if ((board->castle & castles[color_bit*2+i])
                && (!(board->all & castle_masks_1[color_bit*2+i]))){
                    // if the opponent can only move to squares (dsts)
                    // which do not attack the king during castle (mask)
                    // emit the castle
                if (!(dsts & mask)) {
                    EMIT_MOVE(moves, castle_king_pos_before[color_bit], castle_king_pos_after[color_bit*2+i]);
                }
            }
        }
    }

    /*
    if (board->color) {
        moves += gen_black_pawn_moves(board, moves);
    }
    else {
        moves += gen_white_pawn_moves(board, moves);
    }
    */


    return moves - ptr; // incompatible with parallel code, for now it is just for refactoring
}

int gen_legal_moves(Board *board, Move *moves) {
    Move *ptr = moves;
    Undo undo;
    Move temp[MAX_MOVES];
    int count = gen_moves_new(board, temp);
    for (int i = 0; i < count; i++) {
        Move *move = &temp[i];
        do_move(board, move, &undo);
        if (!is_illegal(board)) {
            memcpy(moves++, move, sizeof(Move));
        }
        undo_move(board, move, &undo);
    }
    return moves - ptr;
}
