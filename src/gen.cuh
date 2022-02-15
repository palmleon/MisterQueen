#ifndef GEN_H
#define GEN_H

#include "bb.cuh"
#include "board.h"
#include "move.cuh"
#include "device_launch_parameters.h"

__device__ __host__ int gen_knight_moves(Move *moves, bb srcs, bb mask);
__device__ __host__ int gen_bishop_moves(Move *moves, bb srcs, bb mask, bb all);
__device__ __host__ int gen_rook_moves(Move *moves, bb srcs, bb mask, bb all);
__device__ __host__ int gen_queen_moves(Move *moves, bb srcs, bb mask, bb all);
__device__ __host__ int gen_king_moves(Move *moves, bb srcs, bb mask);

__device__ __host__ int gen_white_pawn_moves(Board *board, Move *moves);
__device__ __host__ int gen_white_knight_moves(Board *board, Move *moves);
__device__ __host__ int gen_white_bishop_moves(Board *board, Move *moves);
__device__ __host__ int gen_white_rook_moves(Board *board, Move *moves);
__device__ __host__ int gen_white_queen_moves(Board *board, Move *moves);
__device__ __host__ int gen_white_king_moves(Board *board, Move *moves);
__device__ __host__ int gen_white_moves(Board *board, Move *moves);

__device__ __host__ int gen_white_pawn_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_knight_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_bishop_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_rook_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_queen_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_king_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_white_attacks(Board *board, Move *moves);
__device__ __host__ int gen_white_checks(Board *board, Move *moves);

__device__ __host__ int gen_black_pawn_moves(Board *board, Move *moves);
__device__ __host__ int gen_black_knight_moves(Board *board, Move *moves);
__device__ __host__ int gen_black_bishop_moves(Board *board, Move *moves);
__device__ __host__ int gen_black_rook_moves(Board *board, Move *moves);
__device__ __host__ int gen_black_queen_moves(Board *board, Move *moves);
__device__ __host__ int gen_black_king_moves(Board *board, Move *moves);
__device__ __host__ int gen_black_moves(Board *board, Move *moves);

__device__ __host__ int gen_black_pawn_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_knight_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_bishop_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_rook_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_queen_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_king_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_attacks_against(Board *board, Move *moves, bb mask);
__device__ __host__ int gen_black_attacks(Board *board, Move *moves);
__device__ __host__ int gen_black_checks(Board *board, Move *moves);

__device__ __host__ int gen_moves(Board *board, Move *moves);
__device__ __host__ int gen_legal_moves(Board *board, Move *moves);
__device__ __host__ int gen_attacks(Board *board, Move *moves);
__device__ __host__ int gen_checks(Board *board, Move *moves);
__device__ __host__ int is_check(Board *board);
__device__ __host__ int is_illegal(Board *board);
__device__ __host__ int has_legal_moves(Board *board);

#endif
