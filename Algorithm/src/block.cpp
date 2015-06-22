#include "block.h"
#include "io.h"
#include "list.h"
#include "tool.h"
#include<algorithm>
#include<cstdio>
#include <omp.h>

int initEdgeBlock(
    const vector< Edge > &edge, int nodeNum, int blockSize,
    EdgeMatrix &block, vector< int > &rowWidth
){
    int blockDim = averageCeil(nodeNum, blockSize);
    vector< int* > blockEdgeNum(blockDim);
    countBlockEdgeNum(edge, blockDim, blockSize, blockEdgeNum);

    int *newID = new int[blockDim];
    rowWidth = vector< int >(blockDim);
    int newBlockDim = integrateBlock(blockEdgeNum, blockDim, newID, rowWidth);
    rowWidth.resize(newBlockDim);
    splitBlock(edge, newID, blockSize, newBlockDim, block);

    for(int i = 0; i < blockDim; i++)
        delete [] blockEdgeNum[i];
    delete [] newID;

    return newBlockDim;
}

void countBlockEdgeNum(
    const vector< Edge > &edge, int blockDim, int blockSize,
    vector< int* > &blockEdgeNum
){
    for(int i = 0; i < blockDim; i++){
        blockEdgeNum[i] = new int [blockDim];
        for(int j = i; j < blockDim; j++){
            blockEdgeNum[i][j] = 0;
        }
    }

    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        int ublock = u / blockSize;
        int vblock = v / blockSize;
        blockEdgeNum[ublock][vblock]++;
    }
}

int integrateBlock(
    const vector< int* > &blockEdgeNum, int blockDim,
    int *newID, vector< int > &rowWidth
){
    int currEdgeNum = blockEdgeNum[0][0];
    int currBlockID = 0;
    int startBlock = 0;
    newID[0] = 0;
    rowWidth[0] = 1;
    for(int b = 1; b < blockDim; b++){
        int addEdgeNum = 0;
        for(int i = startBlock; i <= b; i++)
            addEdgeNum += blockEdgeNum[i][b];
        if(currEdgeNum + addEdgeNum > EDGE_NUM_LIMIT){
            currEdgeNum = blockEdgeNum[b][b];
            newID[b] = ++currBlockID;
            rowWidth[currBlockID] = 1;
            startBlock = b;
        }
        else{
            currEdgeNum += addEdgeNum;
            newID[b] = currBlockID;
            rowWidth[currBlockID]++;
        }
    }
    return currBlockID+1;
}

void splitBlock(
    const vector<Edge> &edge, const int *newID, int blockSize, int blockDim,
    EdgeMatrix &block
){
    block = EdgeMatrix(blockDim);
    for(int i = 0; i < blockDim; i++){
        block[i] = EdgeRow(blockDim);
    }

    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        int ublock = newID[u/blockSize];
        int vblock = newID[v/blockSize];
        block[ublock][vblock].push_back((Edge){u, v});
    }
}

void setEmptyArray(int nodeNum, int *arr){
    #pragma omp parallel for
    for(int i = 0; i <= nodeNum; i++){
        arr[i] = 1;
    }
}

