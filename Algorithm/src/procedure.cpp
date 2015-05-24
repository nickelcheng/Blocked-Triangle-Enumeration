#include "io.h"
#include "solve.h"
#include "timer.h"

#include<cstdio>
#include <cstdlib>
#include <algorithm>

int assignProc, threadNum, blockNum;

int main(int argc, char *argv[]){
    if(argc != 3 && argc != 5){
        fprintf(stderr, "usage: proc <assign_proc> <input_path> <thread_per_block> <block_num>\n");
        return 0;
    }

    extern int assignProc, threadNum, blockNum;
    assignProc = atoi(argv[1]);

    if(assignProc < LIST || assignProc > G_MAT){
        fprintf(stderr, "algo choice\n0: forward\n1: g_forward\n2: mat\n3: g_mat\n");
        return 0;
    }
    if(assignProc == G_LIST || assignProc == G_MAT){
        if(argc != 5){
            fprintf(stderr, "thread_per_block and block_num is needed\n");
            return 0;
        }
        blockNum = atoi(argv[4]);
        threadNum = atoi(argv[3]);
    }

    vector< Edge > edge;
    int nodeNum = inputEdge(argv[2], edge);
    
    timerInit(1)
    timerStart(0)
    std::sort(edge.begin(), edge.end());
    long long triNum = solveBlock(edge, nodeNum);
    timerEnd("time", 0)

    printf("total triangle: %lld\n", triNum);

    return 0;
}
