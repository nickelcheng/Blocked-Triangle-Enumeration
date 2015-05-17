#include "solve.h"
#include "list.h"
#include "mat.h"
#include "timer.h"

long long solveBlock(int blockSize, vector< Edge > &edge, int algo){
    int method;
    if(algo == UNDEF)
        method = scheduler(blockSize, (int)edge.size());
    else method = algo;

    timerInit(1)
    if(method == FORWARD){
        timerStart(0)
        long long triNum = forward(CPU, blockSize, edge);
        timerEnd("forward", 0)
        return triNum;
    }
    else if(method == G_FORWARD){
        int blockNum = 2048;
        int threadNum = 512;
        timerStart(0)
        long long triNum = forward(GPU, blockSize, edge, threadNum, blockNum);
        timerEnd("g_forward", 0)
        return triNum;
    }
    else if(method == MAT){
        long long triNum = mat(CPU, blockSize, edge);
        return triNum;
    }
    else if(method == G_MAT){
        int blockNum = 2048;
        int threadNum = 512;
        long long triNum = mat(GPU, blockSize, edge, threadNum, blockNum);
        return triNum;
    }
    return FORWARD;
}

int scheduler(int nodeNum, int edgeNum){
    return G_FORWARD;
}
