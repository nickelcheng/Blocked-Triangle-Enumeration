#include "solve.h"
#include "list.h"
#include "mat.h"
#include<cstdio>

long long solveBlock(int blockSize, vector< Edge > &edge, int algo){
    int method;
    int blockNum = 1024;
    int threadNum = 512;
    if(algo < FORWARD || algo > G_MAT)
        method = scheduler(blockSize, (int)edge.size());
    else method = algo;

    if(method == FORWARD){
        printf("use forward\n");
        long long triNum = forward(CPU, blockSize, edge);
        return triNum;
    }
    else if(method == G_FORWARD){
        printf("use g_forward\n");
        long long triNum = forward(GPU, blockSize, edge, threadNum, blockNum);
        return triNum;
    }
    else if(method == MAT){
        printf("use mat\n");
        long long triNum = mat(CPU, blockSize, edge);
        return triNum;
    }
    else if(method == G_MAT){
        printf("use g_mat\n");
        long long triNum = mat(GPU, blockSize, edge, threadNum, blockNum);
        return triNum;
    }
    return FORWARD;
}

int scheduler(int nodeNum, int edgeNum){
    return FORWARD;
}
