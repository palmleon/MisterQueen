#include "search.h"
#include "bk.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>


static Board board;
static Search search;

void handle_go(char *line) {
    search.uci = 1;
    /*//search.use_book = 0;
    search.duration = 4;
    char *key;
    char *token = tokenize(line, " ", &key);
    while (token) {
        if (strcmp(token, "infinite") == 0) {
            search.duration = 0;
            //search.use_book = 0;
        }
        else if (strcmp(token, "movetime") == 0) {
            char *arg = tokenize(NULL, " ", &key);
            search.duration = atoi(arg) / 1000.0;
        }
        token = tokenize(NULL, " ", &key);
    }*/
    do_search(&search, &board);
}

void print_menu(void) {
    printf("insert a command\n");
    printf("bm: generate the best move\n");
    printf("bk: do bk tests\n");
    printf("pb: print the board\n");
    printf("q: quit\n");
}

int main(int argc, char **argv) {
    bb_init();
    if (argc == 3) { // load board from file
        if (strcmp(argv[2], "fen") == 0)
            board_load_file_fen(&board, argv[1]);
        else if (strcmp(argv[2], "square") == 0)
            board_load_file_square(&board, argv[1]);
    }
    else
        board_reset(&board); // load the board as in the initial position
    printf("the board is : \n");
    //board_print(&board);
    char command[10];
    while(1) {
        print_menu();
        scanf("%s", command);
        if (strncmp(command, "bm", 2) == 0)
            handle_go(command + 3);
        else if (strncmp(command, "pb", 2) == 0)
            board_print(&board);
        else if (strncmp(command, "bk", 2) == 0)
            bk_tests();
        else if (strncmp(command, "q", 1) == 0)
            return 0;
    }
}

