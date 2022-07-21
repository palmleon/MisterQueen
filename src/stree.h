#ifndef STREE_H
#define STREE_h

#include "board.h"

/* "STree" stands for "Search Tree" */
/* "STNode stands for "Search Tree Node" */

typedef struct _STNode *STNode;
typedef struct _STree *STree;

struct _STNode{
    Board board;
    int score;
    int nchild;
    STNode* children; //array of children, whose len is nchild
};

struct _STree{
    STNode root;
};

/**
 * @brief Create an empty Search Tree
 * 
 * @param nothing
 */
STree STree_init(void);

/**
 * @brief Deallocate a Search Tree
 * 
 * @param tree: the Search Tree to free 
 */
void STree_free(STree tree);

/**
 * @brief Create a new node for the Search Tree
 * 
 * @param board: the board that defines the Search Tree Node
 * @return STNode: the brand new STNode
 */
STNode STNode_init(Board *board);

/**
 * @brief Deallocate a node of the Search Tree
 * 
 * @param node: the Search Tree node to free
 */
void STNode_free(STNode node);

/**
 * @brief Allocate memory for storing the children of a Search Tree node.
 *        However, the children nodes have still to be manually linked to the current node 
 * 
 * @param node: the Search Tree node whose children shall be defined
 * @param nchild: the number of children to allocate
 */
void STNode_init_children(STNode node, int nchild);

/**
 * @brief Deallocate the children of the current Search Tree node and the array that links to them
 * 
 * @param node: the Search Tree node whose children shall be freed
 */
void STNode_free_children(STNode node);

#endif