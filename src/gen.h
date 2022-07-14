#ifndef GEN_H
#define GEN_H

#include "bb.h"
#include "board.h"
#include "move.h"

int is_check(Board *board, char color);
int is_illegal(Board *board);
int gen_moves(Board *board, Move *moves);
int gen_legal_moves(Board *board, Move *moves);

#endif
