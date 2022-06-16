#include "search.h"
#include "bk.h"
#include <string.h>
#include <stdio.h>

#define DEBUG_CMD "bk"


static Board board;
static Search search;

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

int main(void) {
    
    char command[10] = DEBUG_CMD;
    char board_file[100];
    board_reset(&board); // load the board as in the initial position

    bb_init();

    while(1) {
        print_menu();
//        scanf("%s", command);
        if (strncmp(command, "bm", 2) == 0) {
            search.uci = 1;
            do_search(&search, &board);
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
    }
}

