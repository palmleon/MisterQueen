#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"
//#include "util.h"


#define INF 1000000
#define MATE 100000

typedef struct {
    // input
    int uci;
    // output
    Move move;
    // internal
    int nodes;
} Search;

int do_search(Search *search, Board *board);

#endif //SEARCH_H
