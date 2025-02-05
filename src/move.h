#ifndef MOVE_H
#define MOVE_H

#include "bb.h"
#include "board.h"

#define MAX_MOVES 220

typedef struct __align__(4) {
    unsigned char src;
    unsigned char dst;
    unsigned char promotion;
    unsigned char already_executed;
} Move;

typedef struct __align__ (16) {
    bb ep;
    char piece;
    char capture;
    char castle;
} Undo;

/*void make_move(Board *board, Move *move);
void do_null_move(Board *board, Undo *undo);
void undo_null_move(Board *board, Undo *undo);*/
__device__ __host__ void do_move(Board *board, Move *move, Undo *undo);
__device__ __host__ void undo_move(Board *board, Move *move, Undo *undo);
int score_move(Board *board, Move *move);

void move_to_string(Move *move, char *str);
void move_from_string(Move *move, const char *str);
void notate_move(Board *board, Move *move, char *result);
void print_move(Board *board, Move *move);
int parse_move(Board *board, const char *notation, Move *move);

#endif
