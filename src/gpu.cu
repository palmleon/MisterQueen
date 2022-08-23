#include "gpu.h"
#include "gen.h"
#include "board.h"
#include "config.h"
#include "move.h"
#include "search.h"
#include "eval.h"

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
    EMIT_PROMOTION(m, a, b, KNIGHT)

__device__ __forceinline__ void do_move_ab(Board *board, Move *move, Undo *undo) {
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

__device__  __forceinline__ void undo_move_ab(Board *board, Move *move, Undo *undo) {
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

/* To verify if the current player is in check, 
   we generate all the opponent moves 
   and verify if any of them can directly attack the king
   color: player that may be in check
*/
__device__  __forceinline__ int is_check_ab(Board *board, char color, char *shmem){
    
    const unsigned int thr_idx = threadIdx.x;
    bb *dsts_array = (bb*) (shmem);
    
    const int color_bit = (color ^ BLACK) >> 3;
    const bb players_pieces[2] = {board->white, board->black}; // array defined to avoid an if-else
    const bb front_right_mask[2] = {0xfefefefefefefefeL, 0x7f7f7f7f7f7f7f7fL};
    const bb front_left_mask[2] = {0x7f7f7f7f7f7f7f7fL, 0xfefefefefefefefeL};
    const bb own_pieces = players_pieces[color_bit];
    const bb opponent_pieces = players_pieces[color_bit ^ 1];
    bb mask = ~own_pieces;
    const bb mask_pawn = opponent_pieces | board->ep;
    //bb dsts = 0;

    //for(int sq = 0; sq < 64; sq++){
    for(int sq = thr_idx * (64 / THREADS_PER_NODE); sq < (thr_idx + 1) * (64 / THREADS_PER_NODE); sq++){
        bb dsts = 0;
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
                    dsts |= d_BB_KNIGHT[sq] & mask;
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
                    dsts |= d_BB_KING[sq] & mask;
                    break;
                default: // empty piece
                    break;
            }
        }
        dsts_array[sq] = dsts;
    }
    __syncthreads();
    if (thr_idx == 0) {
        for (int sq = 1; sq < 64; sq++){
            dsts_array[0] |= dsts_array[sq];
        }
    }
    __syncthreads();
    //return (dsts & opponent_pieces & board->kings) != (long long) 0;
    return (dsts_array[0] & opponent_pieces & board->kings) != (long long) 0;
}

__device__  __forceinline__ int is_illegal_ab(Board *board, char *shmem){
     return is_check_ab(board, board->color ^ BLACK, shmem);
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
__device__  __forceinline__ int gen_moves_ab(Board *board, Move *moves, char *shmem){
    
    const unsigned int thr_idx = threadIdx.x;

    Move *ptr = (Move *) shmem;

    Move *moves_sh = (Move *) shmem;
    bb *dsts_array = (bb*) ((Move*) (shmem) + MAX_MOVES);
    char *pieces = (char*) (dsts_array + 64);
    int *nmoves = (int*) (pieces + 64);

    const int color_bit = board->color / 8;
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

    //for(int sq = 0; sq < 64; sq++){
    for(int sq = thr_idx * (64 / THREADS_PER_NODE); sq < (thr_idx + 1) * (64 / THREADS_PER_NODE); sq++){
        pieces[sq] = board_get_piece(board, sq);
        char piece = pieces[sq];       
        dsts = 0;
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
                    dsts |= d_BB_KNIGHT[sq] & mask;
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
                    dsts |= d_BB_KING[sq] & mask;
                    break;
                default: // empty piece
                    break;
            }
            /*// Emit all the moves
            while (dsts) {
                int dst;
                POP_LSB(dst, dsts);
                if ((PIECE(piece) == PAWN) && (BIT(dst) & promo[color_bit])){
                    EMIT_PROMOTIONS(moves, sq, dst);
                }
                else {
                    EMIT_MOVE(moves, sq, dst);
                }
            }*/
        }
        dsts_array[sq] = dsts;
    }
    __syncthreads();
    if (thr_idx == 0){
        for (int sq = 0; sq < 64; sq++) {
            // Emit all the moves
            dsts = dsts_array[sq];
            char piece = pieces[sq];
            while (dsts) {
                int dst;
                POP_LSB(dst, dsts);
                if ((PIECE(piece) == PAWN) && (BIT(dst) & promo[color_bit])){
                    EMIT_PROMOTIONS(moves_sh, sq, dst);
                }
                else {
                    EMIT_MOVE(moves_sh, sq, dst);
                }
            }
        }
    }
    __syncthreads();

    // GENERATE CASTLES
    mask = ~opponent_pieces;
    // look for opponent attacks in the squares where the king should move
    //dsts = 0;
    if ((board->castle & castles[color_bit*2]) || board->castle & castles[color_bit*2+1]){
        //for (int sq = 0; sq < 64; sq++){
        for(int sq = thr_idx * (64 / THREADS_PER_NODE); sq < (thr_idx + 1) * (64 / THREADS_PER_NODE); sq++){
            dsts = 0;
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
                        dsts |= a1 | a2;
                        }
                        break;
                    case KNIGHT:
                        dsts |= d_BB_KNIGHT[sq] & mask;
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
                        dsts |= d_BB_KING[sq] & mask;
                        break;
                    default: // empty piece
                        break;
                }
            }
            dsts_array[sq] = dsts;
        }
        __syncthreads();
        if (thr_idx == 0) {
            for (int sq = 1; sq < 64; sq++) {
                dsts_array[0] |= dsts_array[sq];
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
                    if (!(dsts_array[0] & mask)) {
                        EMIT_MOVE(moves_sh, castle_king_pos_before[color_bit], castle_king_pos_after[color_bit*2+i]);
                    }
                }
            }
        }
    }
    if (thr_idx == 0){
        *nmoves = moves_sh - ptr;
    }
    __syncthreads();
    unsigned int nmoves_reg = *nmoves, i = thr_idx % nmoves_reg;
    for (unsigned int cnt = 0; cnt < nmoves_reg ; cnt++){
        moves[i] = ptr[i];
        i = (i+1) % nmoves_reg;
    }

    return nmoves_reg; // incompatible with parallel code, for now it is just for refactoring
}

/*  Iterative Alpha-Beta Pruning on GPU
 *  Args:
 *      boards: boards of the nodes to start from (the grandparent node)
 *      first_moves: moves to generate the 1st generation child from
 *      second_moves: moves to generate the 2nd generation child from
 *      pos_parent: the id of the 1st generation node (the parent)
 *      scores_parent: array storing the scores of the different nodes
 *      depth: depth of the search
 *      alpha_parent, beta_parent: alpha, beta defined from the caller
 *      nmoves: number of the 2nd generation nodes
 *      baseIdx: base index to compute the thread idx from
 */
__device__ __forceinline__ void alpha_beta_gpu_iter(Board *board_parent, Move *first_moves, Move *second_moves, int *pos_parent, int *scores_parent, int depth, int alpha_parent, int beta_parent, int nmoves, int baseIdx, char *shmem)
{
    //unsigned int idx = baseIdx + threadIdx.x;
    unsigned int node_idx = baseIdx + blockIdx.x;
    //printf("Node: %d", node_idx);
    unsigned int thr_idx = threadIdx.x;

    Board board = *board_parent;
    Move first_move, second_move;
    Move moves[MAX_MOVES * (MAX_DEPTH_PAR)+1];
    int can_move[MAX_DEPTH_PAR + 1] = {0};
    Undo first_undo, second_undo;
    Undo undo[MAX_DEPTH_PAR + 1];
    int scores[MAX_DEPTH_PAR + 1];
    int alpha[MAX_DEPTH_PAR + 1];
    int beta[MAX_DEPTH_PAR + 1];
    int beta_reached[MAX_DEPTH_PAR + 1] = {0};
    int curr_depth = depth;
    int count = 0, board_illegal, curr_idx = -1;

    // Check whether the thread is not in excess
    if (nmoves > 0 && node_idx >= nmoves) { return; }

    // Creating the initial board
    if (first_moves != NULL) {
        if (pos_parent != NULL)
            first_move = first_moves[pos_parent[node_idx]];
        else
            first_move = first_moves[node_idx];
        do_move_ab(&board, &first_move, &first_undo);
    }

    if (second_moves != NULL) {
        second_move = second_moves[node_idx];
        do_move_ab(&board, &second_move, &second_undo);
    }

    alpha[curr_depth] = alpha_parent;
    beta[curr_depth] = beta_parent;

    if (is_illegal_ab(&board, shmem))
    {
        scores[curr_depth] = INF;
    }
    else if (curr_depth == 0)
    {
        scores[curr_depth] = evaluate(&board);
    }
    else
    {
        count = gen_moves_ab(&board, &(moves[0]), shmem);
        curr_idx = count - 1;
    }

    while (curr_idx >= 0)
    {
        count = -1;
        board_illegal = 0;
        while (count != 0 && !moves[curr_idx].already_executed)
        {
            count = 0;
            curr_depth--;
            do_move_ab(&board, &(moves[curr_idx]), &undo[curr_depth]);
            can_move[curr_depth] = 0;
            beta_reached[curr_depth] = 0;
            alpha[curr_depth] = -beta[curr_depth + 1];
            beta[curr_depth] = -alpha[curr_depth + 1];
            board_illegal = is_illegal_ab(&board, shmem);
            if (curr_depth > 0 && !board_illegal)
            {
                count = gen_moves_ab(&board, &(moves[curr_idx + 1]), shmem);
                curr_idx += count;
            }
        }
        // terminal nodes
        if (count == 0 && board_illegal)
        {
            scores[curr_depth] = INF;
        }
        else if (curr_depth == 0 || count == 0)
        {
            scores[curr_depth] = evaluate(&board);
        }

        if (curr_idx >= 0)
        {
            undo_move_ab(&board, &(moves[curr_idx]), &(undo[curr_depth]));
            curr_depth++;
            curr_idx--;
        }
        // at this point, we should have the correct value of scores[curr_depth]
        if (-scores[curr_depth - 1] > -INF)
        {
            can_move[curr_depth] = 1;
        }
        if (-scores[curr_depth - 1] >= beta[curr_depth])
        {
            scores[curr_depth] = beta[curr_depth];
            beta_reached[curr_depth] = 1;
            while (!moves[curr_idx].already_executed)
            {
                curr_idx--;
            }
        }
        else if (-scores[curr_depth - 1] > alpha[curr_depth])
        {
            alpha[curr_depth] = -scores[curr_depth - 1];
        }
        // all children executed, time to check... checks
        if (moves[curr_idx].already_executed || curr_idx < 0)
        {
            if (!beta_reached[curr_depth])
            {
                scores[curr_depth] = alpha[curr_depth];
                if (!can_move[curr_depth])
                {
                    if (is_check_ab(&board, board.color, shmem))
                    {
                        scores[curr_depth] = -MATE;
                    }
                    else
                    {
                        scores[curr_depth] = 0;
                    }
                }
            }
        }
    }
    scores_parent[node_idx] = scores[curr_depth];
    
}

__global__ void alpha_beta_gpu_kernel(Board *board, Move *first_moves, Move *second_moves, int *pos_parent, int *scores, int depth, int alpha, int beta, int count, int baseIdx)
{
    extern __shared__ char shmem[];
    alpha_beta_gpu_iter(board, first_moves, second_moves, pos_parent, scores, depth, alpha, beta, count, baseIdx, shmem);
}

__global__ void alpha_beta_gpu_kernel(Board *board, Move *first_moves, int depth, int alpha, int beta, int *scores)
{
    extern __shared__ char shmem[];
    alpha_beta_gpu_iter(board, first_moves, NULL, NULL, scores, depth, alpha, beta, -1, 0, shmem);
}