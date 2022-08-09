#ifndef BOARD_H
#define BOARD_H

#include "bb.h"

//#define WHITE 0x00
#define WHITE 0x0
//#define BLACK 0x10
#define BLACK 0x8


#define EMPTY 0x0
#define PAWN 0x1
#define KNIGHT 0x2
#define BISHOP 0x3
#define ROOK 0x4
#define QUEEN 0x5
#define KING 0x6

#define WHITE_PAWN 0x1
#define WHITE_KNIGHT 0x2
#define WHITE_BISHOP 0x3
#define WHITE_ROOK 0x4
#define WHITE_QUEEN 0x5
#define WHITE_KING 0x6

/*
#define BLACK_PAWN 0x11
#define BLACK_KNIGHT 0x12
#define BLACK_BISHOP 0x13
#define BLACK_ROOK 0x14
#define BLACK_QUEEN 0x15
#define BLACK_KING 0x16
*/

#define BLACK_PAWN 0x9
#define BLACK_KNIGHT 0xa
#define BLACK_BISHOP 0xb
#define BLACK_ROOK 0xc
#define BLACK_QUEEN 0xd
#define BLACK_KING 0xe

//#define PIECE(x) ((x) & 0x0f)
//#define COLOR(x) ((x) & 0x10)

#define PIECE(x) ((x) & 0x7)
#define COLOR(x) ((x) & 0x8)

#define CASTLE_ALL 15
#define CASTLE_WHITE 3
#define CASTLE_BLACK 12
#define CASTLE_WHITE_KING 1
#define CASTLE_WHITE_QUEEN 2
#define CASTLE_BLACK_KING 4
#define CASTLE_BLACK_QUEEN 8

#define MATERIAL_PAWN 100
#define MATERIAL_KNIGHT 320
#define MATERIAL_BISHOP 330
#define MATERIAL_ROOK 500
#define MATERIAL_QUEEN 900
#define MATERIAL_KING 20000

//const int piece_material[7] = {0, MATERIAL_PAWN, MATERIAL_KNIGHT, MATERIAL_BISHOP, MATERIAL_ROOK, MATERIAL_QUEEN, MATERIAL_KING};

extern const int POSITION_WHITE_PAWN[64];
extern const int POSITION_WHITE_KNIGHT[64];
extern const int POSITION_WHITE_BISHOP[64];
extern const int POSITION_WHITE_ROOK[64];
extern const int POSITION_WHITE_QUEEN[64];
extern const int POSITION_WHITE_KING[64];
extern const int POSITION_BLACK_PAWN[64];
extern const int POSITION_BLACK_KNIGHT[64];
extern const int POSITION_BLACK_BISHOP[64];
extern const int POSITION_BLACK_ROOK[64];
extern const int POSITION_BLACK_QUEEN[64];
extern const int POSITION_BLACK_KING[64];

typedef struct {
    unsigned char squares[32]; // -> valutare se passare a char[32], con quattro pezzi per elemento dato che un pezzo è grande 4 bit
    char color;
    char castle;
    //int white_material;
    //int black_material;
    int material; // material score, >0 for white advantage, <0 for black advantage
    //int white_position;
    //int black_position;
    int position; // position score, >0 for white advantage, <0 for black advantage
    bb ep;  // used for "en-passant" captures
    bb all; // each bit represents a square, if at '1' it's occupied
    bb white; // same as the "all" bitmap, but only with white pieces
    bb black; // same as the "all" bitmap, but only with black pieces
    //bb white_pawns; // same as the "all" bitmap, but only with white pawns // merge the bbs related to the same piece
    //bb black_pawns; // same as the "all" bitmap, but only with black pawns
    bb pawns;
    //bb white_knights; // and so on
    //bb black_knights;
    bb knights;
    //bb white_bishops;
    //bb black_bishops;
    bb bishops;
    //bb white_rooks;
    //bb black_rooks;
    bb rooks;
    //bb white_queens;
    //bb black_queens;
    bb queens;
    //bb white_kings;
    //bb black_kings;
    bb kings;
} Board;

void board_clear(Board *board);
void board_reset(Board *board);
char board_get_piece(Board *board, int sq);
void board_set_piece(Board *board, int sq, char piece);
void board_set(Board *board, int sq, char piece);
void board_print(Board *board);
void board_load_fen(Board *board, char *fen);
void board_load_file_fen(Board *board, char *filename);
void board_load_file_square(Board *board, char *filename);

#endif
