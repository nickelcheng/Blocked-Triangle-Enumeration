#include "io.h"
#include "solve.h"
#include "timer.h"

#include<cstdio>
#include <cstdlib>

int main(int argc, char *argv[]){
    if(argc != 4 && argc != 6){
        fprintf(stderr, "usage: proc <assign_proc> <input_path> <node_num> <thread_per_block> <block_num>\n");
        return 0;
    }

    int nodeNum = atoi(argv[3]);
    int assignProc = atoi(argv[1]);
    int threadNum = 256, blockNum = 1024;

    if(assignProc < LIST || assignProc > G_MAT){
        fprintf(stderr, "algo choice\n0: forward\n1: g_forward\n2: mat\n3: g_mat\n");
        return 0;
    }
    if(assignProc == G_LIST || assignProc == G_MAT){
        if(argc != 6){
            fprintf(stderr, "thread_per_block and block_num is needed\n");
            return 0;
        }
        blockNum = atoi(argv[5]);
        threadNum = atoi(argv[4]);
    }

    vector< Edge > edge;
    inputEdge(argv[2], edge);
    
    timerInit(1)
    timerStart(0)
    long long triNum = solveBlock(edge, nodeNum, assignProc, blockNum, threadNum);
    timerEnd("time", 0)

    printf("total triangle: %lld\n", triNum);

    return 0;
}
