#include<cstdio>
#include<cstdlib>
#include "tools.h"
#include "timer.h"
#include "reorder.h"
#include "triangle.h"
#include "io.h"

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: listIntersect <algorithm> <input_path> <node_num>\n");
        return 0;
    }
    int algo = getAlgo(argv[1]);
    if(algo == -1){
        fprintf(stderr, "algorithm should be forward or edge\n");
        return 0;
    }

    timerInit(2)
    timerStart(0)

    int nodeNum = atoi(argv[3]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    timerStart(1)
    inputList(argv[2], edge);
    timerEnd("input", 1)

    timerStart(1)
    reorder(algo, node, edge);
    timerEnd("reordering", 1)

    timerStart(1)
    int triNum = cpuCountTriNum(node);
    printf("total triangle: %d\n", triNum);
    timerEnd("intersection", 1)


    timerEnd("total", 0)

    return 0;
}

