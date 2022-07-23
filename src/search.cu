#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "search.h"
#include "eval.h"
#include "gen.h"
#include "move.h"
#include "stree.h"
#include "util.h"

#define LEN_POSITIONS 3
#define MAX_DEPTH_PAR 1
#define MAX_DEPTH_SEQ 5

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

void initial_sort_moves_new(Board *board, int *positions, int len) {
    Undo undo;
    Move moves[MAX_MOVES];
    int count = gen_moves(board, moves);
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

/* 
 * Search Tree Generation Function
 * Generate all 2nd generation children of the current node
 * It could be parallelized on the direct children, by generating the moves for all siblings nodes in parallel
 * Since max depth = s, max depth of arg node = s - 2
 */
void gen_search_tree(STNode node){
    Undo undo;
    Move moves[MAX_MOVES];
    if (is_illegal(&(node->board)))
        return;
    for (int i = 0; i < node->nchild; i++){ // <-- PARALLELISM
        if (!is_illegal(&(node->children[i]->board))) {
            int count = gen_moves(&(node->children[i]->board), moves);
            STNode_init_children(node->children[i], count);
            for (int j = 0; j < count; j++) { // <-- PARALLELISM (Moves + Board generate insieme nella gpu)
                Board board_tmp = node->children[i]->board;
                do_move(&board_tmp, &(moves[j]), &undo);
                node->children[i]->children[j] = STNode_init(&board_tmp);
            }                                                          
        }
    }
}

/*
 * Search Tree Destruction Function, acts as an inverse of the gen_search_tree function
 * Destroy all 2nd generation children of the current node
 * Assumptions:
 *   - the node is at ply < s - 2 (i.e. it has effectively generated a search sub-tree)
 */
void undo_search_tree(STNode node){
    for (int i = 0; i < node->nchild; i++){
        STNode_free_children(node->children[i]);
    }
}

/*  Iterative Alpha-Beta Pruning on GPU
 *  Args:
 *      boards: boards of the nodes to start from
 *      childIdx, id of the father node, used to access the right d_scores array
 *      depth: depth of the search
 *      alpha_parent, beta_parent: alpha, beta defined from the caller
 *      scores_parent: array storing the scores of the different nodes
 */

__device__ __forceinline__ void alpha_beta_gpu_iter(Board *boards_parent, int depth, int alpha_parent, int beta_parent, int* scores_parent) {
    
    int idx = threadIdx.x; 
    Board board = boards_parent[idx];
    Move moves[MAX_MOVES * (MAX_DEPTH_PAR)];
    int can_move[MAX_DEPTH_PAR+1] = {0};
    Undo undo[MAX_DEPTH_PAR+1];
    int scores[MAX_DEPTH_PAR+1];
    int alpha[MAX_DEPTH_PAR+1];
    int beta[MAX_DEPTH_PAR+1];
    int beta_reached[MAX_DEPTH_PAR+1] = {0};
    int curr_depth = depth;
    int count = 0, board_illegal, curr_idx = -1;
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //moves[0] = moves_parent[idx+1];
    alpha[curr_depth] = alpha_parent;
    beta[curr_depth] = beta_parent;
    //do_move(&board, &(moves_parent[idx+1]), &undo[curr_depth]);
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
    //undo_move(&board, &(moves_parent[idx+1]), &undo[curr_depth]);
    //scores_parent[idx+1] = -scores[curr_depth];
    scores_parent[idx] = -scores[curr_depth];
}

/*
__device__ __forceinline__ void alpha_beta_gpu_iter(Board *board_parent, int depth, int alpha_parent, int beta_parent, Move* moves_parent, int* scores_parent) {
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
*/

/*__global__ void alpha_beta_gpu_kernel(Board *board_parent, int depth, int alpha, int beta, Move* moves_parent, int* scores){
    alpha_beta_gpu_iter(board_parent, depth, alpha, beta, moves_parent, scores);
}
*/

__global__ void alpha_beta_gpu_kernel(Board *boards, int depth, int alpha, int beta, int *scores){
    alpha_beta_gpu_iter(boards, depth, alpha, beta, scores);
}

/*
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
*/

/*
 * Parallel Alpha-Beta on terminal nodes up to depth d, starting at depth s
 * This function provides more accurate scores for the nodes at depth s, so that they can be used by the Sequential Alpha-Beta con CPU,
 * instead of just evaluating the terminal nodes.
 * Assumptions:
 *   the node is not illegal (check in the Sequental Alpha-Beta)
 */
void alpha_beta_parallel(STNode node, int s, int d, int alpha, int beta){
    
    if (s >= 2) {

        Board **boards, **d_boards;
        int **scores, **d_scores;

        int nchildren;
        // Allocate the Board Table
        boards = (Board **) malloc (node->nchild * sizeof(Board*));
        // Allocate the Board Table as a matrix of boards: each kernel will use its own row
        d_boards = (Board **) malloc (node->nchild * sizeof(Board*));
        //cudaMalloc((void ***) &d_boards, node->nchild * sizeof(Board*));

        //For each child move, allocate a row of the Board Table
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0) {
                boards[i] = (Board*) malloc (nchildren * sizeof(Board));
                cudaMalloc(&(d_boards[i]), nchildren * sizeof(Board));
            }
        }

        // Allocate the Score Table
        scores = (int **) malloc (node->nchild * sizeof(int*));
        // Allocate the Score Table as a matrix of scores: each kernel will use its own row
        d_scores = (int **) malloc (node->nchild * sizeof(int*));

        //cudaMalloc(&d_scores, node->nchild * sizeof(int*));
        //For each child move, allocate a row of the Score Table
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0) {
                scores[i] = (int*) malloc (nchildren * sizeof(int));
                cudaMalloc(&(d_scores[i]), nchildren * sizeof(int));
            }
        }

        // Insert the Boards in the Score Table
        for (int i = 0; i < node->nchild; i++){
            for (int j = 0; j < node->children[i]->nchild; j++){
                boards[i][j] = node->children[i]->children[j]->board;
            }
        }

        // Transfer the Boards to the GPU
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0){
                cudaMemcpy(d_boards[i], boards[i], nchildren * sizeof(Board), cudaMemcpyHostToDevice);
            }
        }

        // Launch the Search
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0){
                alpha_beta_gpu_kernel<<<1, node->children[i]->nchild>>>(d_boards[i], d, alpha, beta, d_scores[i]);
            }
        }

        // Retrieve the results
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0){
                cudaMemcpy(scores[i], d_scores[i], nchildren * sizeof(int), cudaMemcpyDeviceToHost);
            }
        }

        // Update the Search Tree
        for (int i = 0; i < node->nchild; i++){
            for (int j = 0; j < node->children[i]->nchild; j++){
                node->children[i]->children[j]->score = scores[i][j];
            }
        }

        // Free the Score Table
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0) {
                cudaFree(d_scores[i]);
                free(scores[i]);
            }
        }
        free(d_scores); free(scores);

        // Free the Board Table
        for (int i = 0; i < node->nchild; i++){
            nchildren = node->children[i]->nchild;
            if (nchildren > 0) {
                cudaFree(d_boards[i]);
                free(boards[i]);
            }
        }
        free(d_boards); free(boards);
    }

    // Special Cases
    else if (s == 1){
        Board *boards, *d_boards;
        int *scores, *d_scores;

        boards = (Board*) malloc (node->nchild * sizeof(Board));
        cudaMalloc(&d_boards, node->nchild * sizeof(Board));
        scores = (int*) malloc (node->nchild * sizeof(int));
        cudaMalloc(&d_scores, node->nchild * sizeof(int));

        for (int i = 0; i < node->nchild; i++){
            boards[i] = node->children[i]->board;
        }
        
        cudaMemcpy(d_boards, boards, node->nchild * sizeof(Board), cudaMemcpyHostToDevice);

        alpha_beta_gpu_kernel<<<1, node->nchild>>>(d_boards, d, alpha, beta, d_scores);

        cudaMemcpy(scores, d_scores, node->nchild * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < node->nchild; i++){
            node->children[i]->score = scores[i];
        }
        
        cudaFree(d_scores);
        free(scores);
        cudaFree(d_boards);
        free(boards);
    }
    else if (s == 0){
        Board *board, *d_board;
        int *score, *d_score;

        board = (Board*) malloc (sizeof(Board));
        cudaMalloc(&d_board, sizeof(Board));
        score = (int*) malloc (sizeof(int));
        cudaMalloc(&d_score, sizeof(int));

        *board = node->board;
        
        cudaMemcpy(d_board, board, sizeof(Board), cudaMemcpyHostToDevice);

        alpha_beta_gpu_kernel<<<1, 1>>>(d_board, d, alpha, beta, d_score);

        cudaMemcpy(score, d_score, sizeof(int), cudaMemcpyDeviceToHost);

        node->score = *score;
        
        cudaFree(d_score);
        free(score);
        cudaFree(d_board);
        free(board);
    }

    
}

void alpha_beta_cpu_new(STNode node, int s, int d, int ply, int alpha, int beta, int isPV, int *positions) {
    
    int result;
    int can_move = 0, beta_reached = 0;
    //int num_multiproc;
    //cudaDeviceGetAttribute(&num_multiproc, cudaDevAttrMultiProcessorCount, 0);
    //printf("Multiprocessors: %d\n", num_multiproc);
    if (is_illegal(&(node->board))) {
        node->score = INF;
        return;
    }
    if (ply >= s) {
        //node->score = evaluate(&(node->board)); //TODO REMOVE ONCE DEBUGGED
        return; // score has already been defined by the parallel search on terminal nodes
    }
    if (ply <= s-2){
        gen_search_tree(node);
    }
    if ( ply == s-2 || ply == 0 && (s == 1 || s == 0)){
        //perform the Parallel Search (it does not create additional nodes, but enriches them with more accurate scores coming from the deeper parallel search)
        alpha_beta_parallel(node, s, d, alpha, beta);    
     }
    if (isPV){
        if (ply < LEN_POSITIONS) {
            STNode tmp = node->children[0];
            node->children[0] = node->children[positions[ply]];
            node->children[positions[ply]] = tmp;
        }
    }
    for (int i = 0; i < node->nchild; i++){
        if (isPV && i == 0) {
             alpha_beta_cpu_new(node->children[i], s, d, ply + 1, -beta, -alpha, 1, positions);
        }
        else {
             alpha_beta_cpu_new(node->children[i], s, d, ply + 1, -beta, -alpha, 0, positions);
        }
        int score = -node->children[i]->score;
        if (score > -INF){
            can_move = 1;
        }
        if (score >= beta) {
            node->score = beta;
            beta_reached = 1;
            break;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    if (!can_move) {
        if (is_check(&(node->board), node->board.color)) {
            node->score = -MATE + ply;
        } else {
            node->score = 0;
        }
    }
    else if (!beta_reached){
        node->score = alpha;
    }
    if (ply <= s-2){
        undo_search_tree(node); // destroy the 2nd generation children of the current node
    }
        /*
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
    */
}

/*

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

*/

int root_search_new(Board *board, int s, int d, int ply, int alpha, int beta, Move *best_move) {
    
    // If the board is illegal, it is pointless to perform the search
    if(is_illegal(board)) {
        *best_move = NOT_MOVE;
        return INF;
    }

    // Create the Search Tree
    STree search_tree = STree_init();
    search_tree->root = STNode_init(board);

    // Define the PV to hopefully follow the best case
    int *positions = (int*) malloc (LEN_POSITIONS * sizeof(int));
    initial_sort_moves_new(board, positions, LEN_POSITIONS);
    
    // Generate the 1st generation children of the root node, if s != 0 (otherwise, the root node is terminal and should be explored on the GPU)
    Undo undo;
    Move tmp, moves[MAX_MOVES];
    int count = 0;

    if (s > 0){
        count = gen_moves(&(search_tree->root->board), moves);
        STNode_init_children(search_tree->root, count);
        for (int i = 0; i < count; i++){
            do_move(board, &(moves[i]), &undo);
            search_tree->root->children[i] = STNode_init(board);
            undo_move(board, &(moves[i]), &undo);
        } 
    }

    // Reorder moves for correct move-score association
    tmp = moves[0];
    moves[0] = moves[positions[0]];
    moves[positions[0]] = tmp;    

    /* Perform the search, using the Alpha Beta Pruning Algorithm
     * The algorithm will update the search_tree, updating the score of all the children nodes */

    alpha_beta_cpu_new(search_tree->root, s, d, ply, alpha, beta, 1, positions);

    // Fetch the best move and store it
    int can_move = 0;
    Move *best = NULL;
    for (int i = 0; i < count; i++) {
        int score = -search_tree->root->children[i]->score;
        if (score > -INF){
            can_move = 1;
        }
        if (score > alpha){
            alpha = score;
            best = &(moves[i]);
        }
    }

    if (!can_move) {
        if (is_check(board, board->color)) {
            alpha = -MATE;
        } else {
            alpha = 0;
        }
        *best_move = NOT_MOVE;
    }
    
    if (best) {
        memcpy(best_move, best, sizeof(Move));
    }

    // Free everything
    free(positions);
    STree_free(search_tree);

    return alpha;
    // PARALLEL ALPHA-BETA + ALPHA-BETA REDUCE
    /*
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
    */
}

int do_search(Search *search, Board *board) {
    struct timespec start, end;
    int result = 1;
    int score = 0;
    const int s = MAX_DEPTH_SEQ;
    const int d = MAX_DEPTH_PAR;
    int lo = INF;
    int hi = INF;
    int alpha = score - lo;
    int beta = score + hi;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //score = root_search(board, depth, 0, alpha, beta, &search->move);
    score = root_search_new(board, s, d, 0, alpha, beta, &(search->move));
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int millis = compute_interval_ms(&start, &end);
    if (search->move.src == NOT_MOVE.src && search->move.dst == NOT_MOVE.dst) {
        printf("Impossible to find a valid move: ");
        if (score == INF) {
            printf("Invalid board\n");
        }
        else if (score == -MATE){
            printf("Checkmate\n");
        }
        else if (score == 0){
            printf("Draw\n");
        }
    }
    else {
        if (search->uci) {
            char move_string[16];
            move_to_string(&search->move, move_string);
            printf("Stats:\n| depth: %d\n| score: %d\n| time: %d ms\n",
                s+d, score, millis);
        }
        if (search->uci) {
            char move_string[16];
            notate_move(board, &search->move, move_string);
            //move_to_string(&search->move, move_string);
            printf("| best move: %s\n", move_string);
        }
    }
    return millis;
}

/*

__device__ int alpha_beta_gpu_device(Board *board, int depth, int ply, int alpha, int beta) {
    int alpha_reg = alpha;
    int result;
    //printf("%lu\n", board->all);
    if (is_illegal(board)) {
        result = INF;
    }
    else if (depth <= 0) {
        result = evaluate(board);
    }
    else {
        Undo undo;
        Move moves[MAX_MOVES];
        int count = gen_moves(board, moves);
        int can_move = 0;
        for (int i = 0; i < count; i++) {
            Move *move = &moves[i];
            do_move(board, move, &undo);
            //int score = 0;
            int score = -alpha_beta_gpu_device2(board, depth - 1, ply + 1, -beta, -alpha_reg);
            undo_move(board, move, &undo);
            if (score > -INF) {
                can_move = 1;
            }
            if (score >= beta) {
                return beta;
            }
            if (score > alpha_reg) {
                alpha_reg = score;
            }
        }
        result = alpha_reg;
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
    //return 0;
} */

/*__global__ void alpha_beta_gpu_kernel(Board *board_parent, int depth, int alpha, int beta, Move* moves_parent, int* scores) {
    //int result;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int alpha_reg = alpha;
    Board board = *board_parent;
    Move moves[MAX_MOVES];
    Undo undo;
    int final_score;
    //if (idx > nthreads)
    //    return;
    //for (int i = 0; i < nmoves; i++){
    do_move(&board, &(moves_parent[idx+1]), &undo);
    //do_move(&board, &(moves_parent[idx]), &undo);
    
    if (is_illegal(&board)) {
        final_score = INF;
    }
    else if (depth <= 0) {
        final_score = evaluate(&board);
    }
    else {
        Undo undo;
        int count = gen_moves(&board, moves);
        int can_move = 0;
        int beta_reached = 0;
        for (int i = 0; i < count && !beta_reached; i++) {
            Move *move = &(moves[i]);
            do_move(&board,  move, &undo);
            //Board *board2 = &board;
            //int score = 0;
            int score = -alpha_beta_gpu_device(&board, depth - 1, ply + 1, -beta, -alpha_reg);
            //int score = -alpha_beta_gpu_device(&board, 0, 1, 1, 1);
            //int score = -alpha_beta_gpu_device(board_parent, depth - 1, ply + 1, -beta, -alpha);
            undo_move(&board, move, &undo);
            if (score > -INF) {
                can_move = 1;
            }
            if (score >= beta) {
                //return beta;
                final_score = beta;
                beta_reached = 1;
                //break;
            }
            else if (score > alpha_reg) {
                alpha_reg = score;
            }
        }
        if (!beta_reached){
            final_score = alpha_reg;
            //result = alpha;
            if (!can_move) {
                if (is_check(&board, board.color)) {
                    //result = -MATE + ply;
                    final_score = -MATE + ply;
                } else {
                    //result = 0;
                    final_score = 0;
                }
            }
        }
    }
    scores[idx+1] = -final_score;
}*/