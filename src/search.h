#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"

#define INF 10000000
#define MATE 1000000
#define LEN_POSITIONS 3
#define MAX_DEPTH_PAR 2
#define MAX_DEPTH_SEQ 4

int do_search(Board *board, int uci, Move *move);

#endif //SEARCH_H
