# (Parallel) Mister Queen

Mr. Queen is a chess engine written in C, originally developed by [Michael Fogleman](https://www.michaelfogleman.com/).

The purpose of this project was to parallelize the original Chess Engine using two different techniques.

## The original engine

The original engine implemented many advanced search techniques, e.g. Iterative Deepening or Transposition Tables, which have been removed because they were not parallelizable at all on a GPU without adding a noticeable synchronization overhead.

## Parallelization Techniques

Two different techniques have been explored:

- PV-split
- "SeqPar" Search

Both the methods try to infer the Principal Variation, i.e. to order the search tree such that the left-most branch contains the best move. This is achieved by executing a sequential search at much lower depth.

Once the PV has been found, the engine executes the Alpha-Beta pruning algorithm to find the best move.

- In PV-split, the PV is executed sequentially, while all the sibling node are launched in parallel
- In SeqPar Search, the whole Search Tree is explored sequentially up to depth s, while the terminal nodes are further explored up to depth d; so the total depth is equal to s+d