#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"
//#include "util.h"

#define INF 10000000
#define MATE 1000000
#define MAX_DEPTH 6
#define THREADS_PER_NODE 16

int do_search(Board *board, int uci, Move *move);

#endif //SEARCH_H
