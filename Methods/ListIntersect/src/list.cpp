#include<cstdio>
#include<cstdlib>
#include "tools.h"
#include "timer.h"
#include "reorder.h"
#include "triangle.h"
#include "io.h"

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: list <algorithm> <input_path> <node_num>\n");
        return 0;
    }
    int algo = getAlgo(argv[1]);
    if(algo == -1){
        fprintf(stderr, "algorithm should be forward or edge\n");
        return 0;
    }

    int nodeNum = atoi(argv[3]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    inputList(argv[2], edge);

    timerInit(1)
    timerStart(0)
    reorder(algo, node, edge);
    int triNum = cpuCountTriNum(node);
    timerEnd("total", 0)

    printf("total triangle: %d\n", triNum);
    return 0;
}

