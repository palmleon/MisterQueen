#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"
//#include "util.h"

#define INF 10000000
#define MATE 1000000
#define LEN_POSITIONS 3
#define MAX_DEPTH_PAR 3
#define MAX_DEPTH_SEQ 3

typedef struct {
    // input
    int uci;
    // output
    Move move;
    // internal
    //int nodes;
} Search;

int do_search(Search *search, Board *board);

#endif //SEARCH_H
