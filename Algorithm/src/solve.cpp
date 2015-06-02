#include "solve.h"
#include "reorder.h"
#include "list.h"
#include "mat.h"
#include "tool.h"
#include<cstdio>
#include<algorithm>
#include "timer.h"

void solveBlock(const vector< Edge > &edge, int blockSize){
    if(edge.empty()) return;

    int entry = averageCeil(blockSize, BIT_PER_ENTRY);
    scheduler(edge, blockSize, edge, blockSize, entry);
}

void mergeBlock(const vector< Matrix > &block, int x, int y, int blockSize){
    if(block[x][y].empty()) return;

    vector< Edge > edge;
    edge.insert(edge.end(), block[x][x].begin(), block[x][x].end());
    edge.insert(edge.end(), block[y][y].begin(), block[y][y].end());
    edge.insert(edge.end(), block[x][y].begin(), block[x][y].end());
    if(edge.empty()) return;

    vector< Edge >::const_iterator e = block[x][y].begin();
    for(int i = 0; e != block[x][y].end(); ++e, i++){
        edge.push_back(Edge(e->v, e->u));
    }
    std::sort(edge.begin(), edge.end());
    
    int entry = averageCeil(2*blockSize, BIT_PER_ENTRY);
    scheduler(block[x][y], blockSize, edge, 2*blockSize, entry);
}

void intersectBlock(const vector< Matrix > &block, int x, int y, int z, int blockSize){
    if(block[x][y].empty()) return;

    vector< Edge > edge;
    edge.insert(edge.end(), block[x][z].begin(), block[x][z].end());
    edge.insert(edge.end(), block[y][z].begin(), block[y][z].end());
    if(edge.empty()) return;

    int entry = averageCeil(blockSize, BIT_PER_ENTRY);
    scheduler(block[x][y], blockSize, edge, 2*blockSize, entry);
}

void scheduler(
    const vector< Edge > &edge, int edgeRange, 
    const vector< Edge > &target, int nodeNum, int entryNum
){
    // todo: auto deside proc, blockNum, and threadNum
    int proc = LIST, threadNum = 256, blockNum = 1024;
    extern int assignProc;
    if(assignProc >= LIST && assignProc <= G_MAT)
        proc = assignProc;
    else
        getStrategy(nodeNum, (int)target.size(), proc);

/*    timerInit(1)
    timerStart(0)*/

    if(proc == LIST){
        list(CPU, edge, edgeRange, target, nodeNum);
    }
    else if(proc == G_LIST){
        list(GPU, edge, edgeRange, target, nodeNum, threadNum, blockNum);
    }
    else if(proc == MAT){
        mat(CPU, edge, edgeRange, target, nodeNum, entryNum);
    }
    else if(proc == G_MAT){
        mat(GPU, edge, edgeRange, target, nodeNum, entryNum, threadNum, blockNum);
    }
//    timerEnd("time", 0)
}

void getStrategy(int nodeNum, int edgeNum, int &proc){
    double density = edgeNum / (nodeNum*(nodeNum-1)) * 100;
    int cpu, gpu;
    if((nodeNum <= 1024 && density < 10) || (nodeNum <= 2048 && density < 7))
        cpu = LIST;
    else
        cpu = MAT;
    if((nodeNum <= 1024 && density < 30) || (nodeNum <= 10240 && density < 20) || nodeNum > 10240)
        gpu = G_LIST;
    else
        gpu = G_MAT;

    if(nodeNum <= 1024 || (nodeNum <= 1536 && density < 20) || (nodeNum <= 2048 && density < 15) || (nodeNum <= 3072 && density < 8) || (nodeNum <= 6144 && density < 4))
        proc = cpu;
    else
        proc = gpu;
}

