#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"

#define INF 10000000
#define MATE 1000000
#define LEN_POSITIONS 3

// Total Search Depth = MAX_DEPTH_SEQ + MAX_DEPTH_PAR
#define MAX_DEPTH_SEQ 4 // MUST BE GREATER THAN 0
#define MAX_DEPTH_PAR 2

int do_search(Board *board, int uci, Move *move);

#endif //SEARCH_H
