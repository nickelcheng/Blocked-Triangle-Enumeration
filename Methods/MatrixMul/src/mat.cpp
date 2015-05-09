#include<cstdio>
#include<cstdlib>
#include "tools.h"
#include "timer.h"
#include "matFunc.h"

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: mat <input_path> <node_num> <node_per_tile>\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    int nodePerTile = atoi(argv[3]);
    if(nodePerTile % BIT_PER_ENTRY != 0){
        fprintf(stderr, "node per tile must be multiple of %lu\n", BIT_PER_ENTRY);
        return 0;
    }

    int entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    UI *mat = (UI*)malloc(entryNum*nodeNum*sizeof(UI));

    inputMat(argv[1], mat, entryNum*nodeNum*sizeof(UI), entryNum);

    timerInit(1)
    timerStart(0)
    int triNum = cpuCountTriNum(nodeNum, nodePerTile, mat);
    timerEnd("total", 0)

    free(mat);

    printf("total triangle: %d\n", triNum);
    return 0;
}

