#include "bb.h"
#include "GPUsearch.h"
#include <stdlib.h>

static Board board;
static Search search;
static thrd_t thrd;

static int thread_func(void *arg) {
    do_search(&search, &board);
    return 0;
}

static void thread_start() {
    thrd_create(&thrd, thread_func, NULL);
}

void handle_go(char *line) {
    search.uci = 1;
    search.use_book = 1;
    search.duration = 4;
    char *key;
    char *token = tokenize(line, " ", &key);
    while (token) {
        if (strcmp(token, "infinite") == 0) {
            search.duration = 0;
            search.use_book = 0;
        }
        else if (strcmp(token, "movetime") == 0) {
            char *arg = tokenize(NULL, " ", &key);
            search.duration = atoi(arg) / 1000.0;
        }
        else if (strcmp(token, "ponder") == 0) {
            return; // no pondering yet
        }
        token = tokenize(NULL, " ", &key);
    }
    thread_start();
}

int main(int argc, char **argv) {
    bb_init();
    handle_go("");
    return 0;
}

