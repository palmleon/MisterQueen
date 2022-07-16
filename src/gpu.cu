#include <stdio.h>
#include "bb.h"
#include "board.h"
#include "move.h"
#include "search.h"
#include "gpu.h"

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

/* EVALUATION */

__device__ __forceinline__ int count_stacked_pawns_gpu(bb pawns, int count) {
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

__device__ __forceinline__ int evaluate_gpu(Board *board) {
    int score = 0;
    // evaluate the total score square by square
    score += board->material;
    // evaluate position, square by square
    score += board->position;

    // evaluate stacked pawns
    score -= count_stacked_pawns_gpu(board->pawns & board->white, 2) * 50;
    score -= count_stacked_pawns_gpu(board->pawns & board->white, 3) * 100;
    score += count_stacked_pawns_gpu(board->pawns & board->black, 2) * 50;
    score += count_stacked_pawns_gpu(board->pawns & board->black, 3) * 100;
    return board->color ? -score : score;
}

/* BOARD UPDATE */

// Updates also the score of the board (score_move only computes it on the fly)
__device__ __forceinline__ char board_get_piece_gpu(Board *board, int sq) {
    int sq_shift[2] = {4,0};
    char sq_masks[2] = {0xf0, 0x0f};
    unsigned char before_shift = board->squares[sq/2] & sq_masks[sq % 2];
    char after_shift = before_shift >> sq_shift[sq % 2];
    return after_shift;
}

__device__ __forceinline__ void board_set_piece_gpu(Board *board, int sq, char piece) {
    int sq_shift[2] = {4,0};
    char sq_masks[2] = {0x0f, 0xf0};
    board->squares[sq/2] = (piece << sq_shift[sq % 2]) | (board->squares[sq/2] & sq_masks[sq % 2]);
}

__device__ __forceinline__ void board_set_gpu(Board *board, int sq, char piece) {   
    const int materials[6] = {MATERIAL_PAWN, MATERIAL_KNIGHT, MATERIAL_BISHOP, MATERIAL_ROOK, MATERIAL_QUEEN, MATERIAL_KING};
    const int coeff[2] = {1, -1};
    #ifdef __CUDA_ARCH__
    const int* position_tables[12] = {d_POSITION_WHITE_PAWN, d_POSITION_WHITE_KNIGHT, d_POSITION_WHITE_BISHOP, d_POSITION_WHITE_ROOK, d_POSITION_WHITE_QUEEN, d_POSITION_WHITE_KING, d_POSITION_BLACK_PAWN, d_POSITION_BLACK_KNIGHT, d_POSITION_BLACK_BISHOP, d_POSITION_BLACK_ROOK, d_POSITION_BLACK_QUEEN, d_POSITION_BLACK_KING};
    #else
    const int* position_tables[12] = {POSITION_WHITE_PAWN, POSITION_WHITE_KNIGHT, POSITION_WHITE_BISHOP, POSITION_WHITE_ROOK, POSITION_WHITE_QUEEN, POSITION_WHITE_KING, POSITION_BLACK_PAWN, POSITION_BLACK_KNIGHT, POSITION_BLACK_BISHOP, POSITION_BLACK_ROOK, POSITION_BLACK_QUEEN, POSITION_BLACK_KING};
    #endif
    
    bb* piece_masks[6] = {&(board->pawns), &(board->knights), &(board->bishops), &(board->rooks), &(board->queens), &(board->kings)};
    //bb* piece_masks[12] = {&(board->white_pawns), &(board->white_knights), &(board->white_bishops), &(board->white_rooks), &(board->white_queens), &(board->white_kings), &(board->black_pawns), &(board->black_knights), &(board->black_bishops), &(board->black_rooks), &(board->black_queens), &(board->black_kings)};
    bb* color_masks[2] = {&(board->white), &(board->black)};
    //int sq_shift[2] = {4,0};
    //char sq_masks[2] = {0x0f, 0xf0};
    //char previous = board->squares[sq]; // take the previous piece on that square
    unsigned char previous = board_get_piece_gpu(board, sq);
    board_set_piece_gpu(board, sq, piece);
    //const char color_previous = COLOR(previous) >> 4;
    //const char color_piece = COLOR(piece) >> 4;
    /*printf("%d %d %d %d\n", COLOR(previous), COLOR(previous) >> 1, COLOR(previous) >> 2, COLOR(previous) >> 3);
    printf("color: %d\n", COLOR(previous));
    printf("piece: %d\n", PIECE(previous));
    const unsigned char color = COLOR(previous);
    */
    const unsigned char color_previous = COLOR(previous) / 8; //>> 3;// >> 3;
    const unsigned char color_piece = COLOR(piece) / 8; //>> 3;
    if (previous) { // was the square empty?
        // There was sth before: remove the previous piece
        bb mask = ~BIT(sq);
        board->all &= mask; // bitwise removal of the piece
 //       printf("Board->all: %lu\n", board->all);
        board->material -= materials[PIECE(previous)-1]*coeff[color_previous];
//        printf("Board->material: %lu\n", board->material);
        board->position -= position_tables[color_previous*6+(PIECE(previous)-1)][sq] * coeff[color_previous];
//        printf("Board->position: %lu\n", board->position);
        
        //*(piece_masks[PIECE(previous)-1]) &= mask;
        //*(piece_masks[color_previous*6+PIECE(previous)-1]) &= mask;

        *(piece_masks[PIECE(previous)-1]) &= mask;
        *(color_masks[color_previous]) &= mask;
    }
    if (piece) { // if the piece to move exists (is the if necessary?)
        bb bit = BIT(sq); // place it
        board->all |= bit;
        board->material += materials[PIECE(piece)-1]*coeff[color_piece];
        board->position += position_tables[color_piece*6+(PIECE(piece)-1)][sq]*coeff[color_piece];
        //*(piece_masks[PIECE(piece)-1]) |= bit;
        //*(piece_masks[color_piece*6+PIECE(piece)-1]) |= bit;
        *(piece_masks[PIECE(piece)-1]) |= bit;
        *(color_masks[color_piece]) |= bit;
    }
}

/* MOVE GENERATION */
__device__ __forceinline__ int is_check_gpu(Board *board, char color){
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
        char piece = board_get_piece_gpu(board, sq);
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

__device__ __forceinline__ int is_illegal_gpu(Board *board){
    return is_check_gpu(board, board->color ^ BLACK);
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
__device__ __forceinline__ int gen_moves_gpu(Board *board, Move *moves, void* shmem){
    
    const int idy = threadIdx.y;
    // declare different pointers to the same contiguous memory area: it is the only way to use the shared memory
    bb *dsts_array = (bb*) shmem;
    int *pieces = (int*) ((char*) shmem + 64*sizeof(bb));
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
    bb dsts; 
    __syncthreads();
    for(int sq = idy * (64 / THREADS_PER_NODE); sq < (idy + 1) * (64 / THREADS_PER_NODE); sq++){
        //char piece = board->squares[sq];
        //char piece = board_get_piece_gpu(board, sq);
        pieces[sq] = board_get_piece_gpu(board, sq);
        char piece = pieces[sq];
        dsts = 0;
        //dsts_array[sq] = 0;
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
                    dsts = bb_bishop(sq, board->all) & mask;
                    break;
                case ROOK:
                    //dsts_array[sq] = bb_rook(sq, board->all) & mask;
                    dsts = bb_rook(sq, board->all) & mask;
                    break;
                case QUEEN:
                    //dsts_array[sq] = bb_queen(sq, board->all) & mask;
                    dsts = bb_queen(sq, board->all) & mask;
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
    // GENERATE CASTLES
    mask = ~opponent_pieces;
    // look for opponent attacks in the squares where the king should move
    //dsts = 0;
    if ((board->castle & castles[color_bit*2]) || board->castle & castles[color_bit*2+1]){
        //for (int sq = 0; sq < 64; sq++){
        __syncthreads();
        for(int sq = idy * (64 / THREADS_PER_NODE); sq < (idy + 1) * (64 / THREADS_PER_NODE); sq++){
            //char piece = board_get_piece_gpu(board, sq);
            //dsts = 0;
            char piece = pieces[sq];
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
                        dsts /*|*/= bb_bishop(sq, board->all) & mask;
                        //dsts_array[sq] |= bb_bishop(sq, board->all) & mask;
                        break;
                    case ROOK:
                        dsts /*|*/= bb_rook(sq, board->all) & mask;
                        //dsts_array[sq] |= bb_rook(sq, board->all) & mask;
                        break;
                    case QUEEN:
                        dsts /*|*/= bb_queen(sq, board->all) & mask;
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
        if (idy == 0) {
            for (int sq = 1; sq < 64; sq++) {
                dsts_array[0] |= dsts_array[sq];
            }
        }
        __syncthreads();
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
                if (!(dsts_array[0] & mask)) {
                    EMIT_MOVE(moves, castle_king_pos_before[color_bit], castle_king_pos_after[color_bit*2+i]);
                }
            }
        }
    }

    return moves - ptr; // incompatible with parallel code, for now it is just for refactoring
}

/* MOVE EXECUTION */

__device__ __forceinline__  void do_move_gpu(Board *board, Move *move, Undo *undo) {
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
    undo->piece = board_get_piece_gpu(board, move->src);
    undo->capture = board_get_piece_gpu(board, move->dst);
    undo->castle = board->castle;
    undo->ep = board->ep;
    move->already_executed = 1;
    Move move2 = *move;
    // remove the moving piece from its starting position
    board_set_gpu(board, move->src, EMPTY);
    // define the ending position of the piece
    if (move->promotion) {
        board_set_gpu(board, move->dst, move->promotion | board->color);
    }
    else {
        board_set_gpu(board, move->dst, undo->piece);
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
            board_set_gpu(board, move->dst - coeff[color_bit]*8, EMPTY);
         }
    }
    // check for a castle
    else if (PIECE(undo->piece) == KING) {
        board->castle &= ~castles_all[color_bit];
        for (int i = 0; i < 2; i++) {
            if (move->src == king_start[color_bit] && move->dst == king_arrival[color_bit*2+i]){
                board_set_gpu(board, rook_start[2*color_bit+i], EMPTY);
                board_set_gpu(board, rook_arrival[2*color_bit+i], rooks[color_bit]);
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

__device__ __forceinline__ void undo_move_gpu(Board *board, Move *move, Undo *undo) {
    //TOGGLE_HASH(board);
    const int coeff[2] = {1, -1};
    const int color_bit = (board->color / 8) ^ 1;
    const char pawns[2] = {WHITE_PAWN, BLACK_PAWN};
    const char rooks[2] = {WHITE_ROOK, BLACK_ROOK};
    const int king_start[2] = {4, 60};
    const int king_arrival[4] = {6,2,62,58};
    const int rook_start[4] = {7,0,63,56};
    const int rook_arrival[4] = {5,3,61,59};
    
    board_set_gpu(board, move->src, undo->piece);
    board_set_gpu(board, move->dst, undo->capture);
    board->castle = undo->castle;
    board->ep = undo->ep;
    if (PIECE(undo->piece) == PAWN){
        if (BIT(move->dst) == undo->ep){
            board_set_gpu(board, move->dst - coeff[color_bit]*8, pawns[color_bit^1]);
        }
    }
    else if (PIECE(undo->piece) == KING){
        for (int i = 0; i < 2; i++){
            if (move->src == king_start[color_bit] && move->dst == king_arrival[2*color_bit+i]){
                board_set_gpu(board, rook_start[2*color_bit+i], rooks[color_bit]);
                board_set_gpu(board, rook_arrival[2*color_bit+i], EMPTY);
            }
        }
    }
    board->color ^= BLACK;
}

// ITERATIVE (Terminal) ALPHA-BETA PRUNING

__device__ __forceinline__ void alpha_beta_gpu_iter(Board *board_parent, int depth, int alpha_parent, int beta_parent, Move* moves_parent, int* scores_parent, void *shmem) {
    
    /* Info about the thread */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y;
    //bb *dsts = &(dsts_array[idy]);
    //char *piece_array = shmem + sizeof(bb) * THREADS_PER_NODE;
    //char *piece = &(piece_array[idy]);

    /* Info about Alpha-Beta */
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
    
    //printf("BlockIdx.x: %d; BlockIdx.y: %d; BlockIdx.z: %d; ThreadIdx.x: %d; ThreadIdx.y: %d; ThreadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    //printf("Idx: %d; Idy: %d\n", idx, idy);
    //moves[0] = moves_parent[idx+1];
    alpha[curr_depth] = alpha_parent;
    beta[curr_depth] = beta_parent;
    do_move_gpu(&board, &(moves_parent[idx+1]), &undo[curr_depth]);
    // initial board created
    if(is_illegal_gpu(&board)){
        scores[curr_depth] = INF;
    }
    else if (curr_depth == 0) {
        scores[curr_depth] = evaluate_gpu(&board);
    }
    else {
        count = gen_moves_gpu(&board, &(moves[0]), shmem);
        curr_idx = count - 1;
    }

    while (curr_idx >= 0){ 
        count = -1;
        board_illegal = 0;
        while(count != 0 && !moves[curr_idx].already_executed) {
            count = 0;
            curr_depth--;
            do_move_gpu(&board, &(moves[curr_idx]), &undo[curr_depth]);
            can_move[curr_depth] = 0;
            beta_reached[curr_depth] = 0;
            alpha[curr_depth] = -beta[curr_depth+1];
            beta[curr_depth] = -alpha[curr_depth+1];
            board_illegal = is_illegal_gpu(&board);
            if (curr_depth > 0 && !board_illegal) {
                count = gen_moves_gpu(&board, &(moves[curr_idx+1]), shmem);
                curr_idx += count;
            }
        }
        //terminal nodes
        if (count == 0 && board_illegal) {
            scores[curr_depth] = INF;
        }
        else if (curr_depth == 0 || count == 0) { 
            scores[curr_depth] = evaluate_gpu(&board); 
        }

        if (curr_idx >= 0) {
            undo_move_gpu(&board, &(moves[curr_idx]), &(undo[curr_depth]));
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
            while (!moves[curr_idx].already_executed){ // Pruning
                curr_idx--;
            }
        }
        else if (-scores[curr_depth-1] > alpha[curr_depth]) {
            alpha[curr_depth] = -scores[curr_depth-1];
        }
        // all children executed, time to check... checks
        if (moves[curr_idx].already_executed || curr_idx < 0){  
            if(!beta_reached[curr_depth]){
                scores[curr_depth] = alpha[curr_depth];
                if(!can_move[curr_depth]){
                    if(is_check_gpu(&board, board.color)){
                        scores[curr_depth] = -MATE;
                    }
                    else {
                        scores[curr_depth] = 0;
                    }
                }
            }
        }
    }
    undo_move_gpu(&board, &(moves_parent[idx+1]), &undo[curr_depth]);
    if (idy == 0) {
        scores_parent[idx+1] = -scores[curr_depth];
    }
}

__global__ void alpha_beta_gpu_kernel(Board *board_parent, int depth, int alpha, int beta, Move* moves_parent, int* scores){
    extern __shared__ char shmem[];
    alpha_beta_gpu_iter(board_parent, depth, alpha, beta, moves_parent, scores, (void *) shmem);
}