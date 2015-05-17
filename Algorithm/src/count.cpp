#include<cstdio>
#include<cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "tool.h"
#include "timer.h"

int main(int argc, char *argv[]){
    if(argc != 4 && argc != 5){
        fprintf(stderr, "usage: count <input_path> <node_num> <block_size> (<algo>)\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    int blockSize = atoi(argv[3]);
    int blockDim = averageCeil(nodeNum, blockSize);
    int algo = UNDEF;
    if(argc == 5) algo = atoi(argv[4]);

    vector< Edge > edge;
    vector< Matrix > block(blockDim);

    inputEdge(argv[1], edge);
//    printf("input done\n");
    initBlock(blockDim, block);
//    printf("init block done\n");

    timerInit(2)
    timerStart(0)

    forwardReorder(nodeNum, edge);
    splitBlock(blockSize, block, edge);
    relabelBlock(blockSize, blockDim, block);

/*    for(int i = 0; i < blockDim; i++){
        for(int j = i; j < blockDim; j++){
            printf("block[%d][%d]:", i, j);
            vector< Edge >::iterator it = block[i][j].begin();
            for(; it != block[i][j].end(); ++it){
                printf(" (%d,%d)", it->u, it->v);
            }
            printf("\n");
        }
    }*/

    long long triNum = 0;
    for(int i = 0; i < blockDim; i++){
//        printf("solve block[%d][%d]\n", i, i);
        timerStart(1)
        triNum += solveBlock(blockSize, block[i][i], algo);
        timerEnd("sm", 1)
    }

/*    for(int i = 0; i < blockDim; i++){
        for(int j = i+1; j < blockDim; j++){
            triNum += combine(i, j);
        }
    }*/

    timerEnd("total", 0)

    printf("total triangle: %lld\n", triNum);
    return 0;
}
