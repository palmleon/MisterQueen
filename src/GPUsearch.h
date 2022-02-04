#ifndef PROJECT_GPUSEARCH_H
#define PROJECT_GPUSEARCH_H

#include "board.h"
#include "table.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "eval.h"
#include "gen.h"
#include "move.h"
#include "table.h"
#include "util.h"

#define INF 1000000
#define MATE 100000

typedef struct {
    // input
    int uci;
    int use_book;
    double duration;
    // output
    Move move;
    // control
    int stop;
    // internal
    int nodes;
    Table table;
    PawnTable pawn_table;
} Search;

int do_search(Search *search, Board *board);

#endif //PROJECT_GPUSEARCH_H
