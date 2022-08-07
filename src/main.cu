#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include "bk.h"
#include "config.h"
#include "search.h"
#include "util.h"

#define DEBUG
#define DEBUG_CMD "bk"
#define DEBUG_BOARD ""


void print_menu(void) {
    printf("-------------------\n");
    printf("insert a command:\n");
    printf("square: load a board in square format\n");
    printf("fen: load a board in fen format\n");
    printf("bm: generate the best move\n");
    printf("bk: do bk tests\n");
    printf("pb: print the board\n");
    printf("q: quit\n");
    printf("-------------------\n");
}

void transfer_tables_to_gpu(void) {
    checkCudaErrors(cudaMemcpyToSymbol(d_BB_KNIGHT, BB_KNIGHT, 64 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_BB_KING, BB_KING, 64 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_BB_BISHOP_6, BB_BISHOP_6, 64 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_BB_ROOK_6, BB_ROOK_6, 64 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_MAGIC_BISHOP, MAGIC_BISHOP, 64 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_MAGIC_ROOK, MAGIC_ROOK, 64 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_SHIFT_BISHOP, SHIFT_BISHOP, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_SHIFT_ROOK, SHIFT_ROOK, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_OFFSET_BISHOP, OFFSET_BISHOP, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_OFFSET_ROOK, OFFSET_ROOK, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_ATTACK_BISHOP, ATTACK_BISHOP, 5248 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_ATTACK_ROOK, ATTACK_ROOK, 102400 * sizeof(bb)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_WHITE_PAWN, POSITION_WHITE_PAWN, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_WHITE_KNIGHT, POSITION_WHITE_KNIGHT, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_WHITE_BISHOP, POSITION_WHITE_BISHOP, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_WHITE_ROOK, POSITION_WHITE_ROOK, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_WHITE_QUEEN, POSITION_WHITE_QUEEN, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_WHITE_KING, POSITION_WHITE_KING, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_BLACK_PAWN, POSITION_BLACK_PAWN, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_BLACK_KNIGHT, POSITION_BLACK_KNIGHT, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_BLACK_BISHOP, POSITION_BLACK_BISHOP, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_BLACK_ROOK, POSITION_BLACK_ROOK, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_BLACK_QUEEN, POSITION_BLACK_QUEEN, 64 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_POSITION_BLACK_KING, POSITION_BLACK_KING, 64 * sizeof(int)));
    cudaDeviceSynchronize();  
}

int main(void) {
    
    struct timespec start, end;
    Board board;
    Move move;
    char command[10] = DEBUG_CMD;
    char board_file[100] = DEBUG_BOARD;
    board_reset(&board); // load the board as in the initial position

    bb_init();

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    transfer_tables_to_gpu();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  
    printf("Tables transferred to the GPU! (Time: %d ms)\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);

    #ifndef DEBUG
    while(1) {
    #endif
        print_menu();
        #ifndef DEBUG
        scanf("%s", command);
        #endif
        if (strncmp(command, "bm", 2) == 0) {
            do_search(&board, 1, &move);
        }
        else if (strncmp(command, "square", 6) == 0) {
            printf("Board file: ");
            scanf("%s", board_file);
            board_load_file_square(&board, board_file);
            printf("Board loaded!\n");
        }
        else if (strncmp(command, "fen", 3) == 0) {
            printf("Board file: ");
            scanf("%s", board_file);
            board_load_file_fen(&board, board_file);
            printf("Board loaded!\n");
        }
        else if (strncmp(command, "pb", 2) == 0)
            board_print(&board);
        else if (strncmp(command, "bk", 2) == 0)
            bk_tests();
        else if (strncmp(command, "q", 1) == 0)
            return 0;
    #ifndef DEBUG
    }
    #endif
}

