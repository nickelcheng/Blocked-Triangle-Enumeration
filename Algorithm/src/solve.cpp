#include "solve.h"
#include "reorder.h"
#include "list.h"
#include "mat.h"
#include "tool.h"
#include<cstdio>
#include<algorithm>

long long solveBlock(const vector< Edge > &edge, int blockSize){
    if(edge.empty()) return 0;

    int entry = averageCeil(blockSize, BIT_PER_ENTRY);
    return scheduler(edge, blockSize, edge, blockSize, entry);
}

long long mergeBlock(const vector< Matrix > &block, int x, int y, int blockSize){
    if(block[x][y].empty()) return 0;

    vector< Edge > edge;
    edge.insert(edge.end(), block[x][x].begin(), block[x][x].end());
    edge.insert(edge.end(), block[y][y].begin(), block[y][y].end());
    edge.insert(edge.end(), block[x][y].begin(), block[x][y].end());
    if(edge.empty()) return 0;

    vector< Edge >::const_iterator e = block[x][y].begin();
    for(int i = 0; e != block[x][y].end(); ++e, i++){
        edge.push_back(Edge(e->v, e->u));
    }
    std::sort(edge.begin(), edge.end());
    
    int entry = averageCeil(2*blockSize, BIT_PER_ENTRY);
    return scheduler(block[x][y], blockSize, edge, 2*blockSize, entry);
}

long long intersectBlock(const vector< Matrix > &block, int x, int y, int z, int blockSize){
    if(block[x][y].empty()) return 0;

    vector< Edge > edge;
    edge.insert(edge.end(), block[x][z].begin(), block[x][z].end());
    edge.insert(edge.end(), block[y][z].begin(), block[y][z].end());
    if(edge.empty()) return 0;

    int entry = averageCeil(blockSize, BIT_PER_ENTRY);
    return scheduler(block[x][y], blockSize, edge, 2*blockSize, entry);
}

long long scheduler(
    const vector< Edge > &edge, int edgeRange, 
    const vector< Edge > &target, int nodeNum, int entryNum
){
    // todo: auto deside proc, blockNum, and threadNum
    int proc = LIST, threadNum = 256, blockNum = 1024;
    extern int assignProc;
    if(assignProc >= LIST && assignProc <= G_MAT)
        proc = assignProc;

    long long triNum = 0;
    if(proc == LIST){
//        printf("use list\n");
        triNum = list(CPU, edge, edgeRange, target, nodeNum);
    }
    else if(proc == G_LIST){
//        printf("use g_list\n");
        triNum = list(GPU, edge, edgeRange, target, nodeNum, threadNum, blockNum);
    }
    else if(proc == MAT){
//        printf("use mat\n");
        triNum = mat(CPU, edge, edgeRange, target, nodeNum, entryNum);
    }
/*    else if(proc == G_MAT){
//        printf("use g_mat\n");
        triNum = mat(GPU, blockSize, edge, threadNum, blockNum);
    }*/
    return triNum;
}

