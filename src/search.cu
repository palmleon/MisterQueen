#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "config.h"
#include "search.h"
#include "eval.h"
#include "gen.h"
#include "gpu.h"
#include "move.h"
#include "stree.h"
#include "util.h"

// RICORDA CHE IL PUNTEGGIO DELLA MOSSA VIENE VALUTATO NELLA SORT MOVES e integrato nella board_set()

int initial_sort_moves_rec(Board *board, int *positions, int len, int ply, int alpha, int beta)
{
    int result;
    if (is_illegal(board))
    {
        result = INF;
    }
    else if (len <= 0)
    {
        result = evaluate(board);
    }
    else
    {
        Undo undo;
        Move moves[MAX_MOVES];
        int count = gen_moves(board, moves);
        int *best_indexes = (int *)malloc(sizeof(int) * (len - 1));
        int can_move = 0;
        for (int i = 0; i < count; i++)
        {
            Move *move = &moves[i];
            do_move(board, move, &undo);
            int score = -initial_sort_moves_rec(board, best_indexes, len - 1, ply + 1, -beta, -alpha);
            undo_move(board, move, &undo);
            if (score > -INF)
            {
                can_move = 1;
            }
            if (score >= beta)
            {
                return beta;
            }
            if (score > alpha)
            {
                alpha = score;
                positions[0] = i;
                for (int j = 1; j < len; j++)
                {
                    positions[j] = best_indexes[j - 1];
                }
            }
        }
        result = alpha;
        if (!can_move)
        {
            // if (is_check(board)) {
            if (is_check(board, board->color))
            {
                result = -MATE + ply;
            }
            else
            {
                result = 0;
            }
        }
        free(best_indexes);
    }
    return result;
}

void initial_sort_moves(Board *board, int *positions, int len)
{
    Undo undo;
    Move moves[MAX_MOVES];

    if (len <= 0){
        return;
    }

    int count = gen_moves(board, moves);
    int *best_indexes = (int *)malloc(sizeof(int) * (len - 1));
    int best_score = -INF;
    for (int i = 0; i < count; i++)
    {
        Move *move = &(moves[i]);
        do_move(board, move, &undo);
        int score = -initial_sort_moves_rec(board, best_indexes, len - 1, 1, -INF, +INF);
        undo_move(board, move, &undo);
        if (score > best_score)
        {
            best_score = score;
            positions[0] = i;
            for (int j = 1; j < len; j++)
            {
                positions[j] = best_indexes[j - 1];
            }
        }
    }

    free(best_indexes);
}

/*
 * Search Tree Generation Function
 * Generate all 2nd generation children of the current node
 * It could be parallelized on the direct children, by generating the moves for all siblings nodes in parallel
 * Since max depth = s, max depth of arg node = s - 2
 */
int gen_search_tree(STNode node)
{
    Undo undo;
    cudaStream_t *streams;

    int num_multiproc, max_shmem, max_threads_per_block, max_registers;
    const int shmem_per_node = 64 * (sizeof(bb) + sizeof(char));

    int *d_count, *count;
    Board *d_boards, *boards;
    Move *d_moves, *moves;

    if (is_illegal(&(node->board))) { return 0; }
           
    // Get the Max number of threads per block
    checkCudaErrors(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0));

    // Get the Max Size of the Shared Memory
    checkCudaErrors(cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, 0));

    // Get the Max number of Registers Per Block
    checkCudaErrors(cudaDeviceGetAttribute(&max_registers, cudaDevAttrMaxRegistersPerBlock, 0));

    //printf("registers:%d\n", max_registers);

    // Get the Max number of Nodes per Block, based on the available shared memory
    const int max_nodes_per_block_shmem = max_shmem / shmem_per_node;

    // Get the Max number of Nodes per Block, based on the available registers
    const int max_nodes_per_block_regs = max_registers / (128*THREADS_PER_NODE);

    // Effective number of nodes per Block, depending on the structural limitations of the GPU
    const int nodes_per_block = min(max_nodes_per_block_regs, min(max_nodes_per_block_shmem, max_threads_per_block / THREADS_PER_NODE));

    // Get the Necessary Number of Blocks
    const int num_blocks = node->nchild / nodes_per_block + 1;

    // Define the number of streams as the number of blocks
    const int num_streams = num_blocks;

    // Allocate the board vectors (one board per child node)
    boards = (Board*) malloc (num_blocks * nodes_per_block * sizeof(Board));
    checkCudaErrors(cudaMalloc(&d_boards, num_blocks * nodes_per_block * sizeof(Board)));

    // Allocate the Moves vectors (one subvector per child node)
    moves = (Move*) malloc (num_blocks * nodes_per_block * MAX_MOVES * sizeof(Move));
    checkCudaErrors(cudaMalloc(&d_moves, num_blocks * nodes_per_block * MAX_MOVES * sizeof(Move)));

    // Allocate the count vectors (one count per child node)
    count = (int*) malloc (num_blocks * nodes_per_block * sizeof(int));
    checkCudaErrors(cudaMalloc(&d_count, num_blocks * nodes_per_block * sizeof(int)));
        
    // Create the Streams
    streams = (cudaStream_t *) malloc (num_streams * sizeof(cudaStream_t));   
    for (int i = 0; i < num_streams; i++){
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // Prepare the Board Vector
    for (int i = 0; i < node->nchild; i++) {
        boards[i] = node->children[i]->board;
    }

    // Transfer everything to memory (TODO change the update instruction into stream_idx++)
    for (int i = 0, stream_idx = 0; i < node->nchild; i += nodes_per_block, stream_idx = (stream_idx + 1) % num_streams){
        checkCudaErrors(cudaMemcpyAsync(d_boards + i, boards + i, nodes_per_block * sizeof(Board), cudaMemcpyHostToDevice, streams[stream_idx]));
    }

    for (int i = 0, stream_idx = 0; i < node->nchild; i += nodes_per_block, stream_idx = (stream_idx + 1) % num_streams)
        gen_moves_gpu<<<1, dim3(nodes_per_block, THREADS_PER_NODE, 1), nodes_per_block * 64 * (sizeof(bb) + sizeof(char)), streams[stream_idx]>>>(d_boards, d_moves, d_count, i, node->nchild);

    for (int i = 0, stream_idx = 0; i < node->nchild; i += nodes_per_block, stream_idx = (stream_idx + 1) % num_streams) {
        checkCudaErrors(cudaMemcpyAsync(moves + i*MAX_MOVES, d_moves + i*MAX_MOVES, nodes_per_block * MAX_MOVES * sizeof(Move), cudaMemcpyDeviceToHost, streams[stream_idx]));
        checkCudaErrors(cudaMemcpyAsync(count + i, d_count + i, nodes_per_block * sizeof(int), cudaMemcpyDeviceToHost, streams[stream_idx]));
    }

    cudaDeviceSynchronize();

    int nmoves = 0;
    for (int i = 0; i < node->nchild; i++){
        if (count[i] > 0) {
            nmoves += count[i];
            STNode_init_children(node->children[i], count[i]);
            for (int j = 0; j < count[i]; j++){
                Board board_tmp = node->children[i]->board;
                do_move(&board_tmp, &(moves[i*MAX_MOVES + j]), &undo);
                node->children[i]->children[j] = STNode_init(&board_tmp, &(moves[i*MAX_MOVES + j]));
            }
        }
    }
    
    cudaFree(d_count); cudaFree(d_moves); cudaFree(d_boards);
    free(count); free(moves); free(boards);

    for (int i = 0; i < num_streams; i++){
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    free(streams);
    return nmoves;
}

/*
 * Search Tree Destruction Function, acts as an inverse of the gen_search_tree function
 * Destroy all 2nd generation children of the current node
 * Assumptions:
 *   - the node is at ply < s - 2 (i.e. it has effectively generated a search sub-tree)
 */
void undo_search_tree(STNode node)
{
    for (int i = 0; i < node->nchild; i++)
    {
        STNode_free_children(node->children[i]);
    }
}

/*
 * Parallel Alpha-Beta on terminal nodes up to depth d, starting at depth s
 * This function provides more accurate scores for the nodes at depth s, so that they can be used by the Sequential Alpha-Beta con CPU,
 * instead of just evaluating the terminal nodes.
 * Assumptions:
 *   the node is not illegal (check in the Sequental Alpha-Beta)
 */
void alpha_beta_parallel(STNode node, int s, int d, int alpha, int beta, int count)
{
    if (node->nchild == 0 || count == 0) { return; }

    if (s >= 2)
    {
        cudaStream_t *streams;
        int nchildren;

        // Define all inputs and outputs
        int *pos_parent, *d_pos_parent;
        Move *first_moves, *d_first_moves;
        Move *second_moves, *d_second_moves;
        int *scores, *d_scores;
        Board *d_board;

        // Define the number of streams (TODO try different configurations)
        const int num_streams = node->nchild;
        int stream_idx;
        
        // Define the number of nodes per kernel and the effective size of the 2nd gen moves array
        const int nodes_per_stream = count / num_streams + 1;
        const int num_moves = nodes_per_stream * num_streams;

        // Allocate and create the cudaStreams
        streams = (cudaStream_t*) malloc (num_streams * sizeof(cudaStream_t));
        for (int i = 0; i < num_streams; i++){
            checkCudaErrors(cudaStreamCreate(&(streams[i])));
        }

        // Allocate the Board to start from
        checkCudaErrors(cudaMalloc(&d_board, sizeof(Board)));
        
        // Allocate the Move Array containing the 1st generation moves
        first_moves = (Move *)malloc(node->nchild * sizeof(Move));
        checkCudaErrors(cudaMalloc(&d_first_moves, node->nchild * sizeof(Move)));

        // Allocate the Move Array containing the 2nd generation moves
        second_moves = (Move *)malloc(num_moves * sizeof(Move *));
        checkCudaErrors(cudaMalloc(&d_second_moves, num_moves * sizeof(Move)));

        // Allocate the Score Array
        scores = (int *)malloc(num_moves * sizeof(int));
        checkCudaErrors(cudaMalloc(&d_scores, num_moves * sizeof(int)));

        // Allocate the vector containing the father index for all the 2nd generation nodes
        pos_parent = (int *) malloc(num_moves * sizeof(int));
        checkCudaErrors(cudaMalloc(&d_pos_parent, num_moves * sizeof(int)));

        stream_idx = 0;

        // Define both the 1st and 2nd generation moves
        for (int i = 0, pos = 0; i < node->nchild && pos < count; i++)
        {
            first_moves[i] = node->children[i]->move;
            for (int j = 0; j < node->children[i]->nchild; j++)
            {
                pos_parent[pos] = i;
                second_moves[pos] = node->children[i]->children[j]->move;
                pos++;
            }
        }

        // Transfer the node Board to the GPU
        checkCudaErrors(cudaMemcpy(d_board, &(node->board), sizeof(Board), cudaMemcpyHostToDevice));
        
        // Transfer the 1st generation Move Array to the GPU
        checkCudaErrors(cudaMemcpy(d_first_moves, first_moves, node->nchild * sizeof(Move), cudaMemcpyHostToDevice));

        // Transfer the 2nd generation moves and the father pos array to the GPU, using streams
        for (int i = 0; i < num_streams; i++) {
            checkCudaErrors(cudaMemcpyAsync(d_second_moves + i * nodes_per_stream, second_moves + i * nodes_per_stream, nodes_per_stream * sizeof(Move), cudaMemcpyHostToDevice, streams[i]));
            checkCudaErrors(cudaMemcpyAsync(d_pos_parent + i * nodes_per_stream, pos_parent + i * nodes_per_stream, nodes_per_stream * sizeof(Move), cudaMemcpyHostToDevice, streams[i]));
        }
        
        // Launch the Search
        for (int i = 0; i < num_streams; i++) {
            alpha_beta_gpu_kernel<<<1, nodes_per_stream, 0, streams[i]>>>(d_board, d_first_moves, d_second_moves, d_pos_parent, d_scores, d, alpha, beta, count, i*nodes_per_stream);
        }

        // Retrieve the results
        for (int i = 0; i < num_streams; i++) {
            checkCudaErrors(cudaMemcpyAsync(scores + i * nodes_per_stream, d_scores + i * nodes_per_stream, nodes_per_stream * sizeof(Move), cudaMemcpyDeviceToHost, streams[i]));
        }

        cudaDeviceSynchronize();

        // Update the Search Tree
        for (int i = 0, pos = 0; i < node->nchild && pos < count; i++)
        {
            for (int j = 0; j < node->children[i]->nchild; j++)
            {
                node->children[i]->children[j]->score = scores[pos++];
            }
        }

        // Free all the arrays and the board
        free(pos_parent);
        checkCudaErrors(cudaFree(d_pos_parent));
        free(scores);
        checkCudaErrors(cudaFree(d_scores));
        free(second_moves);
        checkCudaErrors(cudaFree(d_second_moves));
        free(first_moves);
        checkCudaErrors(cudaFree(d_first_moves));
        
        checkCudaErrors(cudaFree(d_board));

        // Destroy the Streams
        for (int i = 0; i < num_streams; i++){
            checkCudaErrors(cudaStreamDestroy(streams[i]));
        }
        free(streams);
    }

    // Special Cases
    else if (s == 1)
    {
        Board *d_board;
        Move *moves, *d_moves;
        int *scores, *d_scores;

        checkCudaErrors(cudaMalloc(&d_board, sizeof(Board)));
        if (node->nchild > 0)
        {
            moves = (Move *)malloc(node->nchild * sizeof(Move));
            checkCudaErrors(cudaMalloc(&d_moves, node->nchild * sizeof(Move)));
            scores = (int *)malloc(node->nchild * sizeof(int));
            checkCudaErrors(cudaMalloc(&d_scores, node->nchild * sizeof(int)));
        }

        for (int i = 0; i < node->nchild; i++)
        {
            moves[i] = node->children[i]->move;
        }

        if (node->nchild > 0)
        {
            checkCudaErrors(cudaMemcpy(d_board, &(node->board), sizeof(Board), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_moves, moves, node->nchild * sizeof(Move), cudaMemcpyHostToDevice));

            alpha_beta_gpu_kernel<<<1, node->nchild>>>(d_board, d_moves, d, alpha, beta, d_scores);

            checkCudaErrors(cudaMemcpy(scores, d_scores, node->nchild * sizeof(int), cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < node->nchild; i++)
        {
            node->children[i]->score = scores[i];
        }

        if (node->nchild > 0)
        {
            checkCudaErrors(cudaFree(d_scores));
            free(scores);
            checkCudaErrors(cudaFree(d_moves));
            free(moves);
        }
        checkCudaErrors(cudaFree(d_board));
    }
}

void alpha_beta_cpu(STNode node, int s, int d, int ply, int alpha, int beta, int isPV, int *positions)
{

    int result;
    int can_move = 0, beta_reached = 0;
    if (is_illegal(&(node->board)))
    {
        node->score = INF;
        return;
    }
    if (node->nchild == 0 && ply < s)
    {
        node->score = evaluate(&node->board);
        return;
    }
    if (ply >= s)
    {
        if (d <= 0){
            node->score = evaluate(&node->board);
        }
        return; // score has already been defined by the parallel search on terminal nodes
    }
    int count = 0;
    if (ply <= s - 2)
    {
        count = gen_search_tree(node);
        if (count == 0) {
            int a = 1;
        }
    }
    if (d > 0 && (ply == s - 2 || (ply == 0 && (s == 1 || s == 0))))
    {
        // perform the Parallel Search (it does not create additional nodes, but enriches them with more accurate scores coming from the deeper parallel search)
        alpha_beta_parallel(node, s, d, alpha, beta, count);
    }
    if (isPV)
    {
        if (ply < LEN_POSITIONS)
        {
            STNode tmp = node->children[0];
            node->children[0] = node->children[positions[ply]];
            node->children[positions[ply]] = tmp;
        }
    }
    for (int i = 0; i < node->nchild; i++)
    {
        if (isPV && i == 0)
        {
            alpha_beta_cpu(node->children[i], s, d, ply + 1, -beta, -alpha, 1, positions);
        }
        else
        {
            alpha_beta_cpu(node->children[i], s, d, ply + 1, -beta, -alpha, 0, positions);
        }
        int score = -node->children[i]->score;
        if (score > -INF)
        {
            can_move = 1;
        }
        if (score >= beta)
        {
            node->score = beta;
            beta_reached = 1;
            break;
        }
        if (score > alpha)
        {
            alpha = score;
        }
    }
    if (!beta_reached) {
        node->score = alpha;
    }
    if (!can_move)
    {
        if (is_check(&(node->board), node->board.color))
        {
            node->score = -MATE + ply;
        }
        else
        {
            node->score = 0;
        }
    }
    if (ply <= s - 2)
    {
        undo_search_tree(node); // destroy the 2nd generation children of the current node
    }
}

int root_search(Board *board, int s, int d, int alpha, int beta, Move *best_move)
{

    // If the board is illegal, it is pointless to perform the search
    if (is_illegal(board))
    {
        *best_move = NOT_MOVE;
        return INF;
    }

    // Create the Search Tree
    STree search_tree = STree_init();
    search_tree->root = STNode_init(board, &NOT_MOVE);
    int *positions;

    // Define the PV to hopefully follow the best case
    if (LEN_POSITIONS > 0) {
        positions = (int *)malloc(LEN_POSITIONS * sizeof(int));
        initial_sort_moves(board, positions, LEN_POSITIONS);
    }
    
    // Generate the 1st generation children of the root node, if s != 0 (otherwise, the root node is terminal and should be explored on the GPU)
    Undo undo;
    Move tmp, moves[MAX_MOVES];
    int count = gen_moves(&(search_tree->root->board), moves);
    STNode_init_children(search_tree->root, count);
    for (int i = 0; i < count; i++)
    {
        do_move(board, &(moves[i]), &undo);
        search_tree->root->children[i] = STNode_init(board, &(moves[i]));
        undo_move(board, &(moves[i]), &undo);
    }

    // Reorder moves for correct move-score association
    if (LEN_POSITIONS > 0) {
        tmp = moves[0];
        moves[0] = moves[positions[0]];
        moves[positions[0]] = tmp;
    }

    /* Perform the search, using the Alpha Beta Pruning Algorithm
     * The algorithm will update the search_tree, updating the score of all the children nodes */

    alpha_beta_cpu(search_tree->root, s, d, 0, alpha, beta, 1, positions);

    int result = alpha;
    // Fetch the best move and store it
    int can_move = 0;
    Move *best = NULL;
    for (int i = 0; i < count; i++)
    {
        int score = -search_tree->root->children[i]->score;
        if (score > -INF)
        {
            can_move = 1;
        }
        if (score > result)
        {
            result = score;
            best = &(moves[i]);
        }
    }

    if (!can_move)
    {
        if (is_check(board, board->color))
        {
            result = -MATE;
        }
        else
        {
            result = 0;
        }
        *best_move = NOT_MOVE;
    }

    if (best)
    {
        memcpy(best_move, best, sizeof(Move));
    }

    // Free everything
    if (LEN_POSITIONS > 0)
        free(positions);
    STree_free(search_tree);

    return result;
}

int do_search(Board *board, int uci, Move *move)
{
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
    // score = root_search(board, depth, 0, alpha, beta, &search->move);
    score = root_search(board, s, d, alpha, beta, move);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int millis = compute_interval_ms(&start, &end);
    if (move->src == NOT_MOVE.src && move->dst == NOT_MOVE.dst)
    {
        printf("Impossible to find a valid move: ");
        if (score == INF)
        {
            printf("Invalid board\n");
        }
        else if (score == -MATE)
        {
            printf("Checkmate\n");
        }
        else if (score == 0)
        {
            printf("Draw\n");
        }
    }
    else
    {
        if (uci)
        {
            char move_string[16];
            move_to_string(move, move_string);
            printf("Stats:\n| depth: %d\n| score: %d\n| time: %d ms\n",
                   s + d, score, millis);
            notate_move(board, move, move_string);
            // move_to_string(&search->move, move_string);
            printf("| best move: %s\n", move_string);
        }
    }
    return millis;
}