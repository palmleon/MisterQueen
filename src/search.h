#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "move.h"

#define INF 10000000
#define MATE 1000000

int do_search(Board *board, int uci, Move *move);

#endif //SEARCH_H
