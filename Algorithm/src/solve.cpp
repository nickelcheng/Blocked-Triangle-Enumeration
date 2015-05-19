#include "solve.h"
#include "reorder.h"
#include "list.h"
#include "mat.h"
#include<cstdio>

long long solveBlock(
    const vector< Edge > &edge, int blockSize,
    int assignProc, int blockNum, int threadNum
){
    if(edge.empty()) return 0;
    int proc = scheduler(blockSize, blockSize, assignProc);
    return depatch(proc, edge, blockSize, edge, blockSize, blockNum, threadNum);
}

long long mergeBlock(
    const vector< Matrix > &block, int x, int y, int blockSize,
    int assignProc, int blockNum, int threadNum
){
    vector< Edge > edge;
    edge.insert(edge.end(), block[x][x].begin(), block[x][x].end());
    edge.insert(edge.end(), block[y][y].begin(), block[y][y].end());
    edge.insert(edge.end(), block[x][y].begin(), block[x][y].end());
    if(edge.empty()) return 0;

    vector< Edge >::const_iterator e = block[x][y].begin();
    for(int i = 0; e != block[x][y].end(); ++e, i++){
        edge.push_back(Edge(e->v, e->u));
    }
    
    int proc = scheduler(2*blockSize, (int)edge.size(), assignProc);
    return depatch(proc, block[x][y], blockSize, edge, 2*blockSize, blockNum, threadNum);
}

long long intersectBlock(
    const vector< Matrix > &block, int x, int y, int z, int blockSize,
    int assignProc, int blockNum, int threadNum
){
    vector< Edge > edge;
    edge.insert(edge.end(), block[x][z].begin(), block[x][z].end());
    edge.insert(edge.end(), block[y][z].begin(), block[y][z].end());
    if(edge.empty()) return 0;

    int proc = scheduler(2*blockNum, (int)edge.size(), assignProc);
    return depatch(proc, block[x][y], blockSize, edge, 2*blockSize, blockNum, threadNum);
}

long long depatch(
    int proc,
    const vector< Edge > &edge, int edgeRange, 
    const vector< Edge > &target, int nodeNum, 
    int blockNum, int threadNum
){
    long long triNum = 0;
    if(proc == LIST){
//        printf("use list\n");
        triNum = list(CPU, edge, edgeRange, target, nodeNum);
    }
    else if(proc == G_LIST){
//        printf("use g_list\n");
        triNum = list(GPU, edge, edgeRange, target, nodeNum, threadNum, blockNum);
    }
/*    else if(proc == MAT){
//        printf("use mat\n");
        triNum = mat(CPU, blockSize, edge);
    }
    else if(proc == G_MAT){
//        printf("use g_mat\n");
        triNum = mat(GPU, blockSize, edge, threadNum, blockNum);
    }*/
    return triNum;
}

int scheduler(int nodeNum, int edgeNum, int assignProc){
    if(assignProc >= LIST && assignProc <= G_MAT)
        return assignProc;
    return LIST;
}

