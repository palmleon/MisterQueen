#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "board.h"

#define MAX_FEN 87
#define ROW_LEN 17

void board_clear(Board *board) {
    memset(board, 0, sizeof(Board));
    board->castle = CASTLE_ALL;
}

/*  Reset the board, set it in the initial configuration */
void board_reset(Board *board) {
    board_clear(board);
    for (int file = 0; file < 8; file++) {
        board_set(board, RF(1, file), WHITE_PAWN);
        board_set(board, RF(6, file), BLACK_PAWN);
    }
    board_set(board, RF(0, 0), WHITE_ROOK);
    board_set(board, RF(0, 1), WHITE_KNIGHT);
    board_set(board, RF(0, 2), WHITE_BISHOP);
    board_set(board, RF(0, 3), WHITE_QUEEN);
    board_set(board, RF(0, 4), WHITE_KING);
    board_set(board, RF(0, 5), WHITE_BISHOP);
    board_set(board, RF(0, 6), WHITE_KNIGHT);
    board_set(board, RF(0, 7), WHITE_ROOK);
    board_set(board, RF(7, 0), BLACK_ROOK);
    board_set(board, RF(7, 1), BLACK_KNIGHT);
    board_set(board, RF(7, 2), BLACK_BISHOP);
    board_set(board, RF(7, 3), BLACK_QUEEN);
    board_set(board, RF(7, 4), BLACK_KING);
    board_set(board, RF(7, 5), BLACK_BISHOP);
    board_set(board, RF(7, 6), BLACK_KNIGHT);
    board_set(board, RF(7, 7), BLACK_ROOK);
}

// Updates also the score of the board (score_move only computes it on the fly)
/* Given a square and a piece, place that piece onto that square */
/*void board_set(Board *board, int sq, char piece) {
    char previous = board->squares[sq]; // take the previous piece on that square
    board->squares[sq] = piece; // place the new piece
    if (previous) { // was the square empty?
        // There was sth before: remove the previous piece
        bb mask = ~BIT(sq);
        board->all &= mask; // bitwise removal of the piece
        if (COLOR(previous)) { // previous piece was black
            board->black &= mask;
            switch (PIECE(previous)) {
                case PAWN:
                    board->black_pawns &= mask;
                    board->black_material -= MATERIAL_PAWN;
                    board->black_position -= POSITION_BLACK_PAWN[sq];
                    //board->hash ^= HASH_BLACK_PAWN[sq];
                    //board->pawn_hash ^= HASH_BLACK_PAWN[sq];
                    break;
                case KNIGHT:
                    board->black_knights &= mask;
                    board->black_material -= MATERIAL_KNIGHT;
                    board->black_position -= POSITION_BLACK_KNIGHT[sq];
                    //board->hash ^= HASH_BLACK_KNIGHT[sq];
                    break;
                case BISHOP:
                    board->black_bishops &= mask;
                    board->black_material -= MATERIAL_BISHOP;
                    board->black_position -= POSITION_BLACK_BISHOP[sq];
                    //board->hash ^= HASH_BLACK_BISHOP[sq];
                    break;
                case ROOK:
                    board->black_rooks &= mask;
                    board->black_material -= MATERIAL_ROOK;
                    board->black_position -= POSITION_BLACK_ROOK[sq];
                    //board->hash ^= HASH_BLACK_ROOK[sq];
                    break;
                case QUEEN:
                    board->black_queens &= mask;
                    board->black_material -= MATERIAL_QUEEN;
                    board->black_position -= POSITION_BLACK_QUEEN[sq];
                   //board->hash ^= HASH_BLACK_QUEEN[sq];
                    break;
                case KING:
                    board->black_kings &= mask;
                    board->black_material -= MATERIAL_KING;
                    board->black_position -= POSITION_BLACK_KING[sq];
                   // board->hash ^= HASH_BLACK_KING[sq];
                   // board->pawn_hash ^= HASH_BLACK_KING[sq];
                    break;
            }
        }
        else {
            board->white &= mask;
            switch (PIECE(previous)) {
                case PAWN:
                    board->white_pawns &= mask;
                    board->white_material -= MATERIAL_PAWN;
                    board->white_position -= POSITION_WHITE_PAWN[sq];
                    //board->hash ^= HASH_WHITE_PAWN[sq];
                    //board->pawn_hash ^= HASH_WHITE_PAWN[sq];
                    break;
                case KNIGHT:
                    board->white_knights &= mask;
                    board->white_material -= MATERIAL_KNIGHT;
                    board->white_position -= POSITION_WHITE_KNIGHT[sq];
                    //board->hash ^= HASH_WHITE_KNIGHT[sq];
                    break;
                case BISHOP:
                    board->white_bishops &= mask;
                    board->white_material -= MATERIAL_BISHOP;
                    board->white_position -= POSITION_WHITE_BISHOP[sq];
                    //board->hash ^= HASH_WHITE_BISHOP[sq];
                    break;
                case ROOK:
                    board->white_rooks &= mask;
                    board->white_material -= MATERIAL_ROOK;
                    board->white_position -= POSITION_WHITE_ROOK[sq];
                   // board->hash ^= HASH_WHITE_ROOK[sq];
                    break;
                case QUEEN:
                    board->white_queens &= mask;
                    board->white_material -= MATERIAL_QUEEN;
                    board->white_position -= POSITION_WHITE_QUEEN[sq];
                   // board->hash ^= HASH_WHITE_QUEEN[sq];
                    break;
                case KING:
                    board->white_kings &= mask;
                    board->white_material -= MATERIAL_KING;
                    board->white_position -= POSITION_WHITE_KING[sq];
                   // board->hash ^= HASH_WHITE_KING[sq];
                   // board->pawn_hash ^= HASH_WHITE_KING[sq];
                    break;
            }
        }
    }
    if (piece) { // if the piece to move exists (is the if necessary?)
        bb bit = BIT(sq); // place it
        board->all |= bit;
        if (COLOR(piece)) {
            board->black |= bit;
            switch (PIECE(piece)) {
                case PAWN:
                    board->black_pawns |= bit;
                    board->black_material += MATERIAL_PAWN;
                    board->black_position += POSITION_BLACK_PAWN[sq];
                    //board->hash ^= HASH_BLACK_PAWN[sq];
                    //board->pawn_hash ^= HASH_BLACK_PAWN[sq];
                    break;
                case KNIGHT:
                    board->black_knights |= bit;
                    board->black_material += MATERIAL_KNIGHT;
                    board->black_position += POSITION_BLACK_KNIGHT[sq];
                    //board->hash ^= HASH_BLACK_KNIGHT[sq];
                    break;
                case BISHOP:
                    board->black_bishops |= bit;
                    board->black_material += MATERIAL_BISHOP;
                    board->black_position += POSITION_BLACK_BISHOP[sq];
                    //board->hash ^= HASH_BLACK_BISHOP[sq];
                    break;
                case ROOK:
                    board->black_rooks |= bit;
                    board->black_material += MATERIAL_ROOK;
                    board->black_position += POSITION_BLACK_ROOK[sq];
                    //board->hash ^= HASH_BLACK_ROOK[sq];
                    break;
                case QUEEN:
                    board->black_queens |= bit;
                    board->black_material += MATERIAL_QUEEN;
                    board->black_position += POSITION_BLACK_QUEEN[sq];
                    //board->hash ^= HASH_BLACK_QUEEN[sq];
                    break;
                case KING:
                    board->black_kings |= bit;
                    board->black_material += MATERIAL_KING;
                    board->black_position += POSITION_BLACK_KING[sq];
                    //board->hash ^= HASH_BLACK_KING[sq];
                    //board->pawn_hash ^= HASH_BLACK_KING[sq];
                    break;
            }
        }
        else {
            board->white |= bit;
            switch (PIECE(piece)) {
                case PAWN:
                    board->white_pawns |= bit;
                    board->white_material += MATERIAL_PAWN;
                    board->white_position += POSITION_WHITE_PAWN[sq];
                    //board->hash ^= HASH_WHITE_PAWN[sq];
                    //board->pawn_hash ^= HASH_WHITE_PAWN[sq];
                    break;
                case KNIGHT:
                    board->white_knights |= bit;
                    board->white_material += MATERIAL_KNIGHT;
                    board->white_position += POSITION_WHITE_KNIGHT[sq];
                    //board->hash ^= HASH_WHITE_KNIGHT[sq];
                    break;
                case BISHOP:
                    board->white_bishops |= bit;
                    board->white_material += MATERIAL_BISHOP;
                    board->white_position += POSITION_WHITE_BISHOP[sq];
                    //board->hash ^= HASH_WHITE_BISHOP[sq];
                    break;
                case ROOK:
                    board->white_rooks |= bit;
                    board->white_material += MATERIAL_ROOK;
                    board->white_position += POSITION_WHITE_ROOK[sq];
                   // board->hash ^= HASH_WHITE_ROOK[sq];
                    break;
                case QUEEN:
                    board->white_queens |= bit;
                    board->white_material += MATERIAL_QUEEN;
                    board->white_position += POSITION_WHITE_QUEEN[sq];
                    //board->hash ^= HASH_WHITE_QUEEN[sq];
                    break;
                case KING:
                    board->white_kings |= bit;
                    board->white_material += MATERIAL_KING;
                    board->white_position += POSITION_WHITE_KING[sq];
                    //board->hash ^= HASH_WHITE_KING[sq];
                    //board->pawn_hash ^= HASH_WHITE_KING[sq];
                    break;
            }
        }
    }
}*/

void board_set(Board *board, int sq, char piece) {
    
    const int materials[6] = {MATERIAL_PAWN, MATERIAL_KNIGHT, MATERIAL_BISHOP, MATERIAL_ROOK, MATERIAL_QUEEN, MATERIAL_KING};
    const int coeff[2] = {1, -1};
    const int* position_tables[12] = {POSITION_WHITE_PAWN, POSITION_WHITE_KNIGHT, POSITION_WHITE_BISHOP, POSITION_WHITE_ROOK, POSITION_WHITE_QUEEN, POSITION_WHITE_KING, POSITION_BLACK_PAWN, POSITION_BLACK_KNIGHT, POSITION_BLACK_BISHOP, POSITION_BLACK_ROOK, POSITION_BLACK_QUEEN, POSITION_BLACK_KING};
    bb* piece_masks[6] = {&(board->pawns), &(board->knights), &(board->bishops), &(board->rooks), &(board->queens), &(board->kings)};
    //bb* piece_masks[12] = {&(board->white_pawns), &(board->white_knights), &(board->white_bishops), &(board->white_rooks), &(board->white_queens), &(board->white_kings), &(board->black_pawns), &(board->black_knights), &(board->black_bishops), &(board->black_rooks), &(board->black_queens), &(board->black_kings)};
    bb* color_masks[2] = {&(board->white), &(board->black)};
    char previous = board->squares[sq]; // take the previous piece on that square
    board->squares[sq] = piece; // place the new piece
    const char color_previous = COLOR(previous) >> 4;
    const char color_piece = COLOR(piece) >> 4;
    if (previous) { // was the square empty?
        // There was sth before: remove the previous piece
        bb mask = ~BIT(sq);
        board->all &= mask; // bitwise removal of the piece
        board->material -= materials[PIECE(previous)-1]*coeff[color_previous];
        board->position -= position_tables[color_previous*6+(PIECE(previous)-1)][sq]*coeff[color_previous];
        //*(piece_masks[PIECE(previous)-1]) &= mask;
        //*(piece_masks[color_previous*6+PIECE(previous)-1]) &= mask;
        *(piece_masks[PIECE(previous)-1]) &= mask;
        *(color_masks[color_previous]) &= mask;
    }
    if (piece) { // if the piece to move exists (is the if necessary?)
        bb bit = BIT(sq); // place it
        board->all |= bit;
        board->material += materials[PIECE(piece)-1]*coeff[color_piece];
        board->position += position_tables[color_piece*6+(PIECE(piece)-1)][sq]*coeff[color_piece];
        //*(piece_masks[PIECE(piece)-1]) |= bit;
        //*(piece_masks[color_piece*6+PIECE(piece)-1]) |= bit;
        *(piece_masks[PIECE(piece)-1]) |= bit;
        *(color_masks[color_piece]) |= bit;
    }
}


/*  Print the board  */
void board_print(Board *board) {
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            char c;
            short int piece = board->squares[RF(rank, file)];
            switch (PIECE(piece)) {
                case EMPTY:  c = '.'; break;
                case PAWN:   c = 'P'; break;
                case KNIGHT: c = 'N'; break;
                case BISHOP: c = 'B'; break;
                case ROOK:   c = 'R'; break;
                case QUEEN:  c = 'Q'; break;
                case KING:   c = 'K'; break;
            };
            if (COLOR(piece)) {
                c |= 0x20;
            }
            putchar(c);
            putchar(' ');
        }
        putchar('\n');
    }
    putchar('\n');
}

/*  Load the whole board, starting from a string whose format can be seen in perft.c */
void board_load_fen(Board *board, char *fen) {
    board_clear(board);
    int i = 0;
    int n = strlen(fen);
    int rank = 7;
    int file = 0;
    for (; i < n; i++) {
        int done = 0;
        switch (fen[i]) {
            case 'P': board_set(board, RF(rank, file++), WHITE_PAWN); break;
            case 'N': board_set(board, RF(rank, file++), WHITE_KNIGHT); break;
            case 'B': board_set(board, RF(rank, file++), WHITE_BISHOP); break;
            case 'R': board_set(board, RF(rank, file++), WHITE_ROOK); break;
            case 'Q': board_set(board, RF(rank, file++), WHITE_QUEEN); break;
            case 'K': board_set(board, RF(rank, file++), WHITE_KING); break;
            case 'p': board_set(board, RF(rank, file++), BLACK_PAWN); break;
            case 'n': board_set(board, RF(rank, file++), BLACK_KNIGHT); break;
            case 'b': board_set(board, RF(rank, file++), BLACK_BISHOP); break;
            case 'r': board_set(board, RF(rank, file++), BLACK_ROOK); break;
            case 'q': board_set(board, RF(rank, file++), BLACK_QUEEN); break;
            case 'k': board_set(board, RF(rank, file++), BLACK_KING); break;
            case '/': file = 0; rank--; break;
            case '1': file += 1; break;
            case '2': file += 2; break;
            case '3': file += 3; break;
            case '4': file += 4; break;
            case '5': file += 5; break;
            case '6': file += 6; break;
            case '7': file += 7; break;
            case '8': file += 8; break;
            case ' ': done = 1; break;
            default: return;
        }
        if (done) {
            if (rank != 0 || file != 8) {
                return;
            }
            break;
        }
    }
    i++;
    switch (fen[i++]) {
        case 'w':
            board->color = WHITE;
            break;
        case 'b':
            board->color = BLACK;
            //board->hash ^= HASH_COLOR;
            //board->pawn_hash ^= HASH_COLOR;
            break;
        default: return;
    }
    i++;
    board->castle = 0;
    for (; i < n; i++) {
        int done = 0;
        switch (fen[i]) {
            case 'K': board->castle |= CASTLE_WHITE_KING; break;
            case 'Q': board->castle |= CASTLE_WHITE_QUEEN; break;
            case 'k': board->castle |= CASTLE_BLACK_KING; break;
            case 'q': board->castle |= CASTLE_BLACK_QUEEN; break;
            case '-': done = 1; break;
            case ' ': done = 1; break;
            default: return;
        }
        if (done) {
            break;
        }
    }
    //board->hash ^= HASH_CASTLE[CASTLE_ALL];
    //board->hash ^= HASH_CASTLE[board->castle];
    //board->pawn_hash ^= HASH_CASTLE[CASTLE_ALL];
    //board->pawn_hash ^= HASH_CASTLE[board->castle];
    i++;
    if (fen[i] == '-') {
        i++;
    }
    else if (fen[i] >= 'a' && fen[i] <= 'h') {
        int ep_file = fen[i] - 'a';
        i++;
        if (fen[i] >= '1' && fen[i] <= '8') {
            int ep_rank = fen[i] - '1';
            board->ep = BIT(RF(ep_rank, ep_file));
            //board->hash ^= HASH_EP[LSB(board->ep) % 8];
            //board->pawn_hash ^= HASH_EP[LSB(board->ep) % 8];
            i++;
        }
    }
    i++;
}

void board_load_file_fen(Board *board, char *filename) {
    FILE* file_ptr = fopen(filename, "r");
    if (file_ptr == NULL) {
        printf("ERROR: error while opening file %s\n", filename);
        printf("Program failed: exit\n");
        exit(EXIT_FAILURE);
    }
    char fen[MAX_FEN];
    fgets(fen, MAX_FEN, file_ptr);
    board_clear(board);
    int i = 0;
    int n = strlen(fen);
    int rank = 7;
    int file = 0;
    for (; i < n; i++) {
        int done = 0;
        switch (fen[i]) {
            case 'P':
                board_set(board, RF(rank, file++), WHITE_PAWN);
                break;
            case 'N':
                board_set(board, RF(rank, file++), WHITE_KNIGHT);
                break;
            case 'B':
                board_set(board, RF(rank, file++), WHITE_BISHOP);
                break;
            case 'R':
                board_set(board, RF(rank, file++), WHITE_ROOK);
                break;
            case 'Q':
                board_set(board, RF(rank, file++), WHITE_QUEEN);
                break;
            case 'K':
                board_set(board, RF(rank, file++), WHITE_KING);
                break;
            case 'p':
                board_set(board, RF(rank, file++), BLACK_PAWN);
                break;
            case 'n':
                board_set(board, RF(rank, file++), BLACK_KNIGHT);
                break;
            case 'b':
                board_set(board, RF(rank, file++), BLACK_BISHOP);
                break;
            case 'r':
                board_set(board, RF(rank, file++), BLACK_ROOK);
                break;
            case 'q':
                board_set(board, RF(rank, file++), BLACK_QUEEN);
                break;
            case 'k':
                board_set(board, RF(rank, file++), BLACK_KING);
                break;
            case '/':
                file = 0;
                rank--;
                break;
            case '1':
                file += 1;
                break;
            case '2':
                file += 2;
                break;
            case '3':
                file += 3;
                break;
            case '4':
                file += 4;
                break;
            case '5':
                file += 5;
                break;
            case '6':
                file += 6;
                break;
            case '7':
                file += 7;
                break;
            case '8':
                file += 8;
                break;
            case ' ':
                done = 1;
                break;
            default:
                return;
        }
        if (done) {
            if (rank != 0 || file != 8) {
                return;
            }
            break;
        }
    }
    i++;
    switch (fen[i++]) {
        case 'w':
            board->color = WHITE;
            break;
        case 'b':
            board->color = BLACK;
            //board->hash ^= HASH_COLOR;
            //board->pawn_hash ^= HASH_COLOR;
            break;
        default: return;
    }
    i++;
    board->castle = 0;
    for (; i < n; i++) {
        int done = 0;
        switch (fen[i]) {
            case 'K': board->castle |= CASTLE_WHITE_KING; break;
            case 'Q': board->castle |= CASTLE_WHITE_QUEEN; break;
            case 'k': board->castle |= CASTLE_BLACK_KING; break;
            case 'q': board->castle |= CASTLE_BLACK_QUEEN; break;
            case '-': done = 1; break;
            case ' ': done = 1; break;
            default: return;
        }
        if (done) {
            break;
        }
    }
    //board->hash ^= HASH_CASTLE[CASTLE_ALL];
    //board->hash ^= HASH_CASTLE[board->castle];
    //board->pawn_hash ^= HASH_CASTLE[CASTLE_ALL];
   // board->pawn_hash ^= HASH_CASTLE[board->castle];
    i++;
    if (fen[i] >= 'a' && fen[i] <= 'h') {
        int ep_file = fen[i] - 'a';
        i++;
        if (fen[i] >= '1' && fen[i] <= '8') {
            int ep_rank = fen[i] - '1';
            board->ep = BIT(RF(ep_rank, ep_file));
            //board->hash ^= HASH_EP[LSB(board->ep) % 8];
            //board->pawn_hash ^= HASH_EP[LSB(board->ep) % 8];
        }
    }
    fclose(file_ptr);
}

/*
 * Load a board from a file; the file contains a board, represented as a squared matrix of pieces
 * Rows go from 8 to 1, column go from A to H, in this order.
 */
void board_load_file_square(Board *board, char *filename) {
    FILE* file_ptr = fopen(filename, "r");
    if (file_ptr == NULL) {
        printf("ERROR: error while opening file %s\n", filename);
        printf("Program failed: exit\n");
        exit(EXIT_FAILURE);
    }
    char line[ROW_LEN];
    int row = 7, column = 0;
    board_clear(board);
    for (int i = 0; i < 8; i++) {
        fgets(line, ROW_LEN, file_ptr);
        for (int j = 0; j < ROW_LEN && line[j] != '\n'; j++) {
            char c = line[j];
            switch (c) {
                case '.':
                    column++;
                    break;
                case 'p':
                    board_set(board, RF(row, column++), BLACK_PAWN);
                    break;
                case 'n':
                    board_set(board, RF(row, column++), BLACK_KNIGHT);
                    break;
                case 'b':
                    board_set(board, RF(row, column++), BLACK_BISHOP);
                    break;
                case 'r':
                    board_set(board, RF(row, column++), BLACK_ROOK);
                    break;
                case 'q':
                    board_set(board, RF(row, column++), BLACK_QUEEN);
                    break;
                case 'k':
                    board_set(board, RF(row, column++), BLACK_KING);
                    break;
                case 'P':
                    board_set(board, RF(row, column++), WHITE_PAWN);
                    break;
                case 'N':
                    board_set(board, RF(row, column++), WHITE_KNIGHT);
                    break;
                case 'B':
                    board_set(board, RF(row, column++), WHITE_BISHOP);
                    break;
                case 'R':
                    board_set(board, RF(row, column++), WHITE_ROOK);
                    break;
                case 'Q':
                    board_set(board, RF(row, column++), WHITE_QUEEN);
                    break;
                case 'K':
                    board_set(board, RF(row, column++), WHITE_KING);
                    break;
                default:
                    break;
            }
        }
        row--;
        column = 0;
    }
    fgets(line, ROW_LEN, file_ptr);
    int i = 0;
    switch (line[i++]) {
        case 'w':
            board->color = WHITE;
            break;
        case 'b':
            board->color = BLACK;
            //board->hash ^= HASH_COLOR;
            //board->pawn_hash ^= HASH_COLOR;
            break;
        default: return;
    }
    board->castle = 0;
    for (; i < ROW_LEN; i++) {
        int done = 0;
        switch (line[i]) {
            case 'K': board->castle |= CASTLE_WHITE_KING; break;
            case 'Q': board->castle |= CASTLE_WHITE_QUEEN; break;
            case 'k': board->castle |= CASTLE_BLACK_KING; break;
            case 'q': board->castle |= CASTLE_BLACK_QUEEN; break;
            case '-': done = 1; break;
            case ' ': break;
            default: return;
        }
        if (done) {
            break;
        }
    }
    //board->hash ^= HASH_CASTLE[CASTLE_ALL];
    //board->hash ^= HASH_CASTLE[board->castle];
    //board->pawn_hash ^= HASH_CASTLE[CASTLE_ALL];
    //board->pawn_hash ^= HASH_CASTLE[board->castle];
    i++;
    if (line[i] >= 'a' && line[i] <= 'h') {
        int ep_file = line[i] - 'a';
        i++;
        if (line[i] >= '1' && line[i] <= '8') {
            int ep_rank = line[i] - '1';
            board->ep = BIT(RF(ep_rank, ep_file));
            //board->hash ^= HASH_EP[LSB(board->ep) % 8];
            //board->pawn_hash ^= HASH_EP[LSB(board->ep) % 8];
        }
    }
    fclose(file_ptr);
}

const int POSITION_WHITE_PAWN[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10,-20,-20, 10, 10,  5,
      5, -5,-10,  0,  0,-10, -5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5,  5, 10, 25, 25, 10,  5,  5,
     10, 10, 20, 30, 30, 20, 10, 10,
     50, 50, 50, 50, 50, 50, 50, 50,
      0,  0,  0,  0,  0,  0,  0,  0,
};

const int POSITION_WHITE_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
};

const int POSITION_WHITE_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
};

const int POSITION_WHITE_ROOK[64] = {
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
};

const int POSITION_WHITE_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
};

const int POSITION_WHITE_KING[64] = {
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
};

const int POSITION_BLACK_PAWN[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
};

const int POSITION_BLACK_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
};

const int POSITION_BLACK_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
};

const int POSITION_BLACK_ROOK[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0,
};

const int POSITION_BLACK_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
};

const int POSITION_BLACK_KING[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
};
