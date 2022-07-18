#include <stdlib.h>
#include "stree.h"
#include "board.h"

/* "STree" stands for "Search Tree" */

/*
 * @brief create an empty tree
 */
STree STree_init(){
    STree tree = (STree) malloc (sizeof(_STree));
    tree->root = NULL; 
    tree->z = NULL;
    return tree;
}

/*
 * @brief free a search tree
 */
void STree_free(STree tree){
    STNode_free(tree->root);
    STNode_free(tree->z);
    free(tree);
}

void STree_set_root(STree tree, STNode root){
    tree->root = root;
}

STNode STree_get_root(STree tree){
    return tree->root;
}

/*
 *  @brief: create a Search Tree Node, with a given board
 */
STNode STNode_init(Board *board){
    STNode node = (STNode) malloc (sizeof(_STNode));
    node->board = *board;
    node->nchild = 0;
    return node;
}

/*
 *  @brief: free a Search Tree Node
 */
void STNode_free(STNode node){
    for (int i = 0; i < node->nchild; i++){
        STNode_free(node->children[i]);
    }
    free(node->children);
    free(node);    
}

/*
 * @brief: define how many children the node has, but do not allocate them yet!
 *         There is no corresponding deallocator function, as STNode_free is sufficient 
 */
void STNode_init_children(STNode node, int nchild){
    node->children = (STNode*) malloc (nchild * sizeof(STNode));
    node->nchild = nchild;
}

void STNode_set_children(STNode node, STNode* children, int nchild){
    node->children = (STNode*) malloc (nchild * sizeof(STNode));
    for (int i = 0; i < nchild; i++){
        node->children[i] = children[i]; //children must have already been allocated
    }
    node->nchild = nchild;
}

STNode *STNode_get_children(STNode node){
    return node->children;
}

void STNode_set_board(STNode node, Board* board){
    node->board = *board;
}

Board STNode_get_board(STNode node){
    return node->board;
}

void STNode_set_nchild(STNode node, int nchild){
    node->nchild = nchild;
}

int STNode_get_nchild(STNode node){
    return node->nchild;
}

void STNode_set_child(STNode node, STNode other, int i){
    node->children[i] = other;
}

STNode STNode_get_child(STNode node, int i){
    return node->children[i];
}

void STNode_free_child(STNode node, int i){
    free(node->children[i]);
}