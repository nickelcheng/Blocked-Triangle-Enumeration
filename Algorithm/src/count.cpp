#include<cstdio>
#include<cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "tool.h"
#include "timer.h"
#include "block.h"
#include <pthread.h>

int assignProc;

int main(int argc, char *argv[]){
    if(argc != 3 && argc != 4){
        fprintf(stderr, "usage: count <input_path> <block_size> (<assign_proc>)\n");
        return 0;
    }

    int blockSize = atoi(argv[2]);
    if(argc == 4) assignProc = atoi(argv[3]);
    else assignProc = UNDEF;

    vector< Edge > edge;

    int nodeNum = inputEdge(argv[1], edge);
    int blockDim = averageCeil(nodeNum, blockSize);
    vector< Matrix > block(blockDim);

    initBlock(blockDim, block);

    timerInit(2)
    timerStart(0)
    timerStart(1)

    forwardReorder(nodeNum, edge);
    splitBlock(blockSize, block, edge);
    sortBlock(block, blockDim);

/*    for(int i = 0; i < blockDim; i++){
        for(int j = i; j < blockDim; j++){
            printf("block[%d][%d]:", i, j);
            printEdge(block[i][j]);
        }
    }*/

    timerEnd("init", 1)
    timerStart(1)

    for(int i = 0; i < blockDim; i++){
        relabelBlock(block[i][i], blockSize, 0, 0);
        for(int j = i+1; j < blockDim; j++){
            relabelBlock(block[i][j], blockSize, 0, 1);
            relabelBlock(block[j][j], blockSize, 1, 1);
//            printf("merge %d & %d\n", i, j);
/*            printBlock(block[i][j], i, j);
            printBlock(block[j][j], j, j);*/
            mergeBlock(block, i, j, blockSize);

            for(int k = j+1; k < blockDim; k++){
                relabelBlock(block[i][k], blockSize, 0, 0);
                relabelBlock(block[j][k], blockSize, 1, 0);
//                printf("intersect %d, base (%d,%d)\n", k, i, j);
/*                printBlock(block[i][k], i, k);
                printBlock(block[j][k], j, k);*/
                intersectBlock(block, i, j, k, blockSize);
            }
        }
    }
    for(int i = 0; i < blockDim; i++){
        relabelBlock(block[i][i], blockSize, 0, 0);
//        printf("solve %d\n", i);
/*        printf("relabel block %d -> (0,0)\n", i);
        printBlock(block[i][i], i, i);*/
        solveBlock(block[i][i], blockSize);
    }

    long long triNum = 0;
    extern vector< pthread_t* > threads;
    vector< pthread_t* >::iterator it;
    for(it = threads.begin(); it != threads.end(); ++it){
        void *ans;
        pthread_join(**it, &ans);
        triNum += *(long long*)ans;
        delete (long long*)ans;
        delete *it;
    }
    timerEnd("solve", 1)

    timerEnd("total", 0)

    printf("total triangle: %lld\n", triNum);
    return 0;
}
