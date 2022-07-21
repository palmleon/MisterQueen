#include <stdlib.h>
#include "stree.h"
#include "board.h"

STree STree_init(void){
    STree tree = (STree) malloc (sizeof(_STree));
    tree->root = NULL;
    return tree;
}

void STree_free(STree tree){
    STNode_free(tree->root);
    free(tree);
}

STNode STNode_init(Board *board){
    STNode node = (STNode) malloc (sizeof(_STNode));
    node->board = *board;
    node->nchild = 0;
    node->score = -INFINITY; //default value
    return node;
}

void STNode_free(STNode node){
    STNode_free_children(node);
    free(node);    
}

void STNode_init_children(STNode node, int nchild){
    node->children = (STNode*) malloc (nchild * sizeof(STNode));
    node->nchild = nchild;
}

void STNode_free_children(STNode node){
    for (int i = 0; i < node->nchild; i++){
        STNode_free(node->children[i]);
    }
    free(node->children);
}

void STNode_set_children(STNode node, STNode* children, int nchild){
    node->children = (STNode*) malloc (nchild * sizeof(STNode));
    for (int i = 0; i < nchild; i++){
        node->children[i] = children[i]; //children must have already been allocated
    }
    node->nchild = nchild;
}