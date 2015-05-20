#include<cstdio>
#include<cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "tool.h"
#include "timer.h"
#include "block.h"

int assignProc;

int main(int argc, char *argv[]){
    if(argc != 4 && argc != 5){
        fprintf(stderr, "usage: count <input_path> <node_num> <block_size> (<assign_proc>)\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    int blockSize = atoi(argv[3]);
    int blockDim = averageCeil(nodeNum, blockSize);
    extern int assignProc;
    if(argc == 5) assignProc = atoi(argv[4]);
    else assignProc = UNDEF;

    vector< Edge > edge;
    vector< Matrix > block(blockDim);

    inputEdge(argv[1], edge);
    initBlock(blockDim, block);

    timerInit(2)
    timerStart(0)

    forwardReorder(nodeNum, edge);
    splitBlock(blockSize, block, edge);
    sortBlock(block, blockDim);

/*    for(int i = 0; i < blockDim; i++){
        for(int j = i; j < blockDim; j++){
            printf("block[%d][%d]:", i, j);
            printEdge(block[i][j]);
        }
    }*/

    long long triNum = 0;
    for(int i = 0; i < blockDim; i++){
        relabelBlock(block[i][i], blockSize, 0, 0);
/*        printf("solve %d\n", i);
        printf("relabel block %d -> (0,0)\n", i);
        printBlock(block[i][i], i, i);*/
        triNum += solveBlock(block[i][i], blockSize);

        for(int j = i+1; j < blockDim; j++){
            relabelBlock(block[i][j], blockSize, 0, 1);
            relabelBlock(block[j][j], blockSize, 1, 1);
/*            printf("merge %d & %d\n", i, j);
            printBlock(block[i][j], i, j);
            printBlock(block[j][j], j, j);*/
            triNum += mergeBlock(block, i, j, blockSize);

            for(int k = j+1; k < blockDim; k++){
                relabelBlock(block[i][k], blockSize, 0, 0);
                relabelBlock(block[j][k], blockSize, 1, 0);
/*                printf("intersect %d, base (%d,%d)\n", k, i, j);
                printBlock(block[i][k], i, k);
                printBlock(block[j][k], j, k);*/
                triNum += intersectBlock(block, i, j, k, blockSize);
            }
        }
    }

    timerEnd("total", 0)

    printf("total triangle: %lld\n", triNum);
    return 0;
}