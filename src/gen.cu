#include <string.h>
#include <math.h>
#include "gen.h"


#define EMIT_MOVE(m, a, b) \
    (m)->src = (a); \
    (m)->dst = (b); \
    (m)->promotion = EMPTY; \
    (m)->already_executed = 0; \
    (m)++;

#define EMIT_PROMOTION(m, a, b, p) \
    (m)->src = (a); \
    (m)->dst = (b); \
    (m)->promotion = (p); \
    (m)->already_executed = 0; \
    (m)++;

#define EMIT_PROMOTIONS(m, a, b) \
    EMIT_PROMOTION(m, a, b, QUEEN) \
    EMIT_PROMOTION(m, a, b, ROOK) \
    EMIT_PROMOTION(m, a, b, BISHOP) \
    EMIT_PROMOTION(m, a, b, KNIGHT)

#define THREADS_PER_NODE 64

/* To verify if the current player is in check, 
   we generate all the opponent moves 
   and verify if any of them can directly attack the king
   color: player that may be in check
*/
__device__ __host__ int is_check(Board *board, char color){
    // for black, board->color >> 4 = 0x01
    // for white, board->color >> 4 = 0x00
    const int color_bit = (color ^ BLACK) >> 3;
    // coeff = -1 for white, +1 for black
    //const int coeff[2] = {-1, 1};
    const bb players_pieces[2] = {board->white, board->black}; // array defined to avoid an if-else
    const bb front_right_mask[2] = {0xfefefefefefefefeL, 0x7f7f7f7f7f7f7f7fL};
    const bb front_left_mask[2] = {0x7f7f7f7f7f7f7f7fL, 0xfefefefefefefefeL};
    const bb own_pieces = players_pieces[color_bit];
    const bb opponent_pieces = players_pieces[color_bit ^ 1];
    bb mask = ~own_pieces;
    const bb mask_pawn = opponent_pieces | board->ep;
    const bb mask_pawn_opp = own_pieces | board->ep;
    bb dsts = 0;

    for(int sq = 0; sq < 64; sq++){
        //char piece = board->squares[sq];
        char piece = board_get_piece(board, sq);
        if (COLOR(piece) >> 3 == color_bit){
            switch(PIECE(piece)){
                case PAWN: {
                    bb pawn_bb = BIT(sq);
                    bb a1 = pawn_bb & front_right_mask[color_bit];
                    bb a1_vec[2] = {a1 << 7, a1 >> 7};
                    a1 = a1_vec[color_bit] & mask_pawn;
                    bb a2 = pawn_bb & front_left_mask[color_bit];
                    bb a2_vec[2] = {a2 << 9, a2 >> 9};
                    a2 = a2_vec[color_bit] & mask_pawn;
                    dsts |= a1;
                    dsts |= a2;      
                }              
                    break;
                case KNIGHT:
                    #ifdef __CUDA_ARCH__
                    dsts |= d_BB_KNIGHT[sq] & mask;
                    #else
                    dsts |= BB_KNIGHT[sq] & mask;
                    #endif
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
                    #ifdef __CUDA_ARCH__
                    dsts |= d_BB_KING[sq] & mask;
                    #else
                    dsts |= BB_KING[sq] & mask;
                    #endif
                    break;
                default: // empty piece
                    break;
            }
        }
    }
    return (dsts & opponent_pieces & board->kings) != (long long) 0;
}

__device__ __host__ int is_illegal(Board *board){
    return is_check(board, board->color ^ BLACK);
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
__device__ __host__ int gen_moves(Board *board, Move *moves){
    Move *ptr = moves;
    // for black, board->color >> 4 = 0x01
    // for white, board->color >> 4 = 0x00
    //const int color_bit = board->color >> 4;
    //const int color_bit = board->color >> 3;
    const int color_bit = board->color / 8;
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
        //char piece = board->squares[sq];
        char piece = board_get_piece(board, sq);
        bb dsts = 0;
        // move a piece only if it is of the current moving player!
        if (COLOR(piece) == board->color){
            bb pawn_bb;
            switch(PIECE(piece)){
                case PAWN: {
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
                    }
                    break;
                case KNIGHT:
                    #ifdef __CUDA_ARCH__
                    dsts |= d_BB_KNIGHT[sq] & mask;
                    #else
                    dsts |= BB_KNIGHT[sq] & mask;
                    #endif
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
                    #ifdef __CUDA_ARCH__
                    dsts |= d_BB_KING[sq] & mask;
                    #else
                    dsts |= BB_KING[sq] & mask;
                    #endif
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
            //char piece = board->squares[sq];
            char piece = board_get_piece(board, sq);
            if (COLOR(piece) != board->color){
                switch(PIECE(piece)){
                    case PAWN: {
                        bb pawn_bb = BIT(sq);
                        bb a1 = pawn_bb & front_right_mask[color_bit ^ 1];
                        bb a1_vec[2] = {a1 << 7, a1 >> 7};
                        a1 = a1_vec[color_bit^1] & mask_pawn_opp;
                        bb a2 = pawn_bb & front_left_mask[color_bit ^ 1];
                        bb a2_vec[2] = {a2 << 9, a2 >> 9};
                        a2 = a2_vec[color_bit^1] & mask_pawn_opp;
                        dsts |= a1 | a2;
                        }
                        break;
                    case KNIGHT:
                        #ifdef __CUDA_ARCH__
                        dsts |= d_BB_KNIGHT[sq] & mask;
                        #else
                        dsts |= BB_KNIGHT[sq] & mask;
                        #endif
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
                        #ifdef __CUDA_ARCH__
                        dsts |= d_BB_KING[sq] & mask;
                        #else
                        dsts |= BB_KING[sq] & mask;
                        #endif
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

    return moves - ptr; // incompatible with parallel code, for now it is just for refactoring
}


__global__ void gen_moves_gpu(Board *board_arr, Move *moves_arr, int *count_arr, int childIdx){
    
    const int idx = threadIdx.x;
    // declare different pointers to the same contiguous memory area: it is the only way to use the shared memory
    __shared__ bb dsts_array[64];
    __shared__ char pieces[64];
    //int *pieces = (int*) ((char*) dsts_array + 64*sizeof(bb));
    Board board = board_arr[childIdx];
    Move *moves = moves_arr + MAX_MOVES * childIdx;
    Move *ptr = moves;
    // for black, board->color >> 4 = 0x01
    // for white, board->color >> 4 = 0x00
    //const int color_bit = board->color >> 4;
    //const int color_bit = board->color >> 3;
    const int color_bit = board.color / 8;
    // coeff = -1 for white, +1 for black
    //const int coeff[2] = {-1, 1};
    const bb players_pieces[2] = {board.white, board.black}; // array defined to avoid an if-else
    const bb promo[2] = {0xff00000000000000L, 0x00000000000000ffL}; // representation of the promotion rank
    const bb third_rank[2] = {0x0000000000ff0000L, 0x0000ff0000000000L}; // used for initial double move of pawn
    const bb front_right_mask[2] = {0xfefefefefefefefeL, 0x7f7f7f7f7f7f7f7fL};
    const bb front_left_mask[2] = {0x7f7f7f7f7f7f7f7fL, 0xfefefefefefefefeL};
    const bb own_pieces = players_pieces[color_bit];
    const bb opponent_pieces = players_pieces[color_bit ^ 1];
    bb mask = ~own_pieces;
    const bb mask_pawn = opponent_pieces | board.ep;
    const bb mask_pawn_opp = own_pieces | board.ep;
    const char castles[4] = {CASTLE_WHITE_KING, CASTLE_WHITE_QUEEN, CASTLE_BLACK_KING, CASTLE_BLACK_QUEEN};
    const bb castle_masks_1[4] = {0x0000000000000060L, 0x000000000000000eL, 0x6000000000000000L, 0x0e00000000000000L};
    const bb castle_masks_2[4] = {0x0000000000000070L, 0x000000000000001cL, 0x7000000000000000L, 0x1c00000000000000L};
    const int castle_king_pos_before[2] = {4, 60};
    const int castle_king_pos_after[4] = {6, 2, 62, 58};
    bb dsts; 
    for(int sq = idx * (64 / THREADS_PER_NODE); sq < (idx + 1) * (64 / THREADS_PER_NODE); sq++){
        //char piece = board->squares[sq];
        //char piece = board_get_piece_gpu(board, sq);
        pieces[sq] = board_get_piece(&board, sq);
        char piece = pieces[sq];
        dsts = 0;
        //dsts_array[sq] = 0;
        // move a piece only if it is of the current moving player!
        if (COLOR(piece) == board.color){
            bb pawn_bb;
            switch(PIECE(piece)){
                case PAWN: {
                    pawn_bb = BIT(sq);
                    bb p1_vec[2] = {pawn_bb << 8, pawn_bb >> 8};
                    bb p1 = p1_vec[color_bit] & ~board.all;
                    bb p2 = p1 & third_rank[color_bit];
                    bb p2_vec[2] = {p2 << 8, p2 >> 8};
                    p2 = p2_vec[color_bit] & ~board.all;
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
                    //dsts_array[sq] |= p1;
                    //dsts_array[sq] |= p2;
                    //dsts_array[sq] |= a1;
                    //dsts_array[sq] |= a2;
                    }
                    break;
                case KNIGHT:
                    #ifdef __CUDA_ARCH__
                    //dsts_array[sq] |= d_BB_KNIGHT[sq] & mask;
                    dsts |= d_BB_KNIGHT[sq] & mask;
                    #else
                    //dsts |= BB_KNIGHT[sq] & mask;
                    #endif
                    break;
                case BISHOP:
                    //dsts_array[sq] = bb_bishop(sq, board->all) & mask;
                    dsts = bb_bishop(sq, board.all) & mask;
                    break;
                case ROOK:
                    //dsts_array[sq] = bb_rook(sq, board->all) & mask;
                    dsts = bb_rook(sq, board.all) & mask;
                    break;
                case QUEEN:
                    //dsts_array[sq] = bb_queen(sq, board->all) & mask;
                    dsts = bb_queen(sq, board.all) & mask;
                    break;
                case KING:
                    #ifdef __CUDA_ARCH__
                    //dsts_array[sq] |= d_BB_KING[sq] & mask;
                    dsts |= d_BB_KING[sq] & mask;                    
                    #else
                    //dsts |= BB_KING[sq] & mask;
                    #endif
                    break;
                default: // empty piece
                    break;
            }
        }
        // define the dsts for square sq in shared memory
        dsts_array[sq] = dsts;
    }
    __syncthreads();
    if (idx == 0) {
        for (int sq = 0; sq < 64; sq++) {
            // Emit all the moves
            dsts = dsts_array[sq];
            char piece = pieces[sq];
            //char piece = board_get_piece_gpu(board, sq);
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
    __syncthreads();

    // GENERATE CASTLES
    mask = ~opponent_pieces;
    // look for opponent attacks in the squares where the king should move
    //dsts = 0;
    if ((board.castle & castles[color_bit*2]) || board.castle & castles[color_bit*2+1]){
        //for (int sq = 0; sq < 64; sq++){
        for(int sq = idx * (64 / THREADS_PER_NODE); sq < (idx + 1) * (64 / THREADS_PER_NODE); sq++){
            //char piece = board_get_piece_gpu(board, sq);
            //dsts = 0;
            char piece = pieces[sq];
            if (COLOR(piece) != board.color){
                switch(PIECE(piece)){
                    case PAWN: {
                        bb pawn_bb = BIT(sq);
                        bb a1 = pawn_bb & front_right_mask[color_bit ^ 1];
                        bb a1_vec[2] = {a1 << 7, a1 >> 7};
                        a1 = a1_vec[color_bit^1] & mask_pawn_opp;
                        bb a2 = pawn_bb & front_left_mask[color_bit ^ 1];
                        bb a2_vec[2] = {a2 << 9, a2 >> 9};
                        a2 = a2_vec[color_bit^1] & mask_pawn_opp;
                        //dsts |= a1 | a2;
                        dsts = a1 | a2;
                        //dsts_array[sq] |= a1 | a2;
                        }
                        break;
                    case KNIGHT:
                        #ifdef __CUDA_ARCH__
                        //dsts |= d_BB_KNIGHT[sq] & mask;
                        dsts = d_BB_KNIGHT[sq] & mask;
                        //dsts_array[sq] |= d_BB_KNIGHT[sq] & mask;
                        #else
                        dsts = BB_KNIGHT[sq] & mask;
                        #endif
                        break;
                    case BISHOP:
                        dsts /*|*/= bb_bishop(sq, board.all) & mask;
                        //dsts_array[sq] |= bb_bishop(sq, board->all) & mask;
                        break;
                    case ROOK:
                        dsts /*|*/= bb_rook(sq, board.all) & mask;
                        //dsts_array[sq] |= bb_rook(sq, board->all) & mask;
                        break;
                    case QUEEN:
                        dsts /*|*/= bb_queen(sq, board.all) & mask;
                        //dsts_array[sq] |= bb_queen(sq, board->all) & mask;
                        break;
                    case KING:
                        #ifdef __CUDA_ARCH__
                        dsts /*|*/= d_BB_KING[sq] & mask;
                        //dsts_array[sq] |= d_BB_KING[sq] & mask;
                        #else
                        dsts |= BB_KING[sq] & mask;
                        #endif
                        break;
                    default: // empty piece
                        break;
                }
            }
            dsts_array[sq] = dsts;
        }
        __syncthreads();
        if (idx == 0) {
            for (int sq = 1; sq < 64; sq++) {
                dsts_array[0] |= dsts_array[sq];
            }
            for (int i = 0; i < 2; i++) {
                // check if the player can castle and, if that is the case,
                // where it can castle and whether there are pieces
                // between the king and the rook
                bb mask = castle_masks_2[color_bit*2+i];
                if ((board.castle & castles[color_bit*2+i])
                    && (!(board.all & castle_masks_1[color_bit*2+i]))){
                        // if the opponent can only move to squares (dsts)
                        // which do not attack the king during castle (mask)
                        // emit the castle
                    if (!(dsts_array[0] & mask)) {
                        EMIT_MOVE(moves, castle_king_pos_before[color_bit], castle_king_pos_after[color_bit*2+i]);
                    }
                }
            }
        }
    }
    if (idx == 0)
        count_arr[childIdx] = moves - ptr;
    return; // incompatible with parallel code, for now it is just for refactoring
}

int gen_legal_moves(Board *board, Move *moves) {
    Move *ptr = moves;
    Undo undo;
    Move temp[MAX_MOVES];
    int count = gen_moves(board, temp);
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
