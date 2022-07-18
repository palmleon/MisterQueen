#ifndef STREE_H
#define STREE_h

#include "board.h"

typedef struct _STNode *STNode;
typedef struct _STree *STree;

struct _STNode{
    Board board;
    int nchild;
    STNode* children; //array of children, whose len is nchild
};

struct _STree{
    STNode root;
    STNode z;
};

STree STree_init();
void STree_free(STree tree);
//void STree_set_root(STree tree, STNode root);
//STNode STree_get_root(STree tree);
STNode STNode_init(Board *board);
void STNode_free(STNode node);
void STNode_init_children(STNode node, int nchild);
//STNode *STNode_get_children(STNode node);
//void STNode_set_board(STNode node, Board board);
//Board STNode_get_board(STNode node);
//void STNode_set_nchild(STNode node, int nchild);
//int STNode_get_nchild(STNode node);
//void STNode_set_child(STNode node, STNode other, int i);
//STNode STNode_get_child(STNode node, int i);
//void STNode_free_child(STNode node, int i);

#endif