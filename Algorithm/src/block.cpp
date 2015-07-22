#include "block.h"
#include "tool.h"
#include <omp.h>
#include <algorithm>
#include <cstdio>

int initEdgeBlock(
    const vector< Edge > &edge, int nodeNum, int blockSize,
    EdgeMatrix &block, vector< int > &rowWidth
){
    int blockDim = averageCeil(nodeNum, blockSize);
    vector< int* > blockEdgeNum(blockDim);
    countBlockEdgeNum(edge, blockDim, blockSize, blockEdgeNum);

    int *newID = new int[blockDim];
    rowWidth = vector< int >(blockDim, 0);
    //int newBlockDim = integrateBlock2(blockEdgeNum, blockDim, newID, rowWidth);
    int newBlockDim = integrateBlock(blockEdgeNum, blockDim, blockSize, newID, rowWidth);
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
    const vector< int* > &blockEdgeNum, int blockDim, int blockSize,
    int *newID, vector< int > &rowWidth
){
    int currEdgeNum = blockEdgeNum[blockDim-1][blockDim-1];
    int currNodeNum = blockSize;
    int currBlockID = blockDim-1;
    int startBlock = blockDim-1;
    newID[blockDim-1] = currBlockID;
    double density = (double)currEdgeNum/((double)blockSize*blockSize/2);
    for(int b = blockDim-2; b >= 0; b--){
        int addEdgeNum = 0;
        for(int i = startBlock; i >= b; i--){
            addEdgeNum += blockEdgeNum[b][i];
        }
        int nodeNum = currNodeNum + blockSize;
        int edgeNum = currEdgeNum + addEdgeNum;
        double newDensity = (double)edgeNum/((double)nodeNum*nodeNum/2);
        if((density > 0.03 && newDensity < 0.03) || edgeNum > EDGE_NUM_LIMIT){
            printf("%d node, %d edge density=%lf, ori=%lf\n", nodeNum, edgeNum, newDensity, density);
            currEdgeNum = blockEdgeNum[b][b];
            currNodeNum = blockSize;
            newID[b] = --currBlockID;
            startBlock = b;
            density = (double)currEdgeNum/((double)blockSize*blockSize/2);
        }
        else{
            currEdgeNum = edgeNum;
            currNodeNum += blockSize;
            newID[b] = currBlockID;
            density = newDensity;
        }
    }
    for(int i = 0; i < blockDim; i++){
        newID[i] -= currBlockID;
        rowWidth[newID[i]]++;
    }
    return blockDim-currBlockID;
}

int integrateBlock2(
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
        arr[i] = 0;
    }
}

void cTransBlock(vector< Edge > &edge, int nodeNum, int uOffset, int vOffset, ListArray &listArr){
    int edgeNum = (int)edge.size();
    listArr.initArray(nodeNum, edgeNum);
    if(edgeNum == 0){
        setEmptyArray(nodeNum, listArr.nodeArr);
        return;
    }
    std::sort(edge.begin(), edge.end());
    cRelabelBlock(uOffset, vOffset, edge);
    cEdge2ListArr(edge, listArr);
}

void cRelabelBlock(int uOffset, int vOffset, vector< Edge > &edge){
    if(uOffset == 0 && vOffset == 0) return;
    int edgeNum = (int)edge.size();
//    #pragma omp parallel for
    for(int i = 0; i < edgeNum; i++){
        edge[i].u -= uOffset;
        edge[i].v -= vOffset;
    }
}

void cEdge2ListArr(const vector< Edge > &edge, ListArray &listArr){
    int nodeNum = listArr.nodeNum;
    int edgeNum = listArr.edgeNum;

//    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++)
        listArr.nodeArr[i] = -1;

//    #pragma omp parallel for
    for(int i = 0; i < edgeNum-1; i++){
        listArr.edgeArr[i] = edge[i].v;
        if(edge[i].u != edge[i+1].u){
            listArr.nodeArr[edge[i+1].u] = i+1;
        }
    }
    listArr.edgeArr[edgeNum-1] = edge[edgeNum-1].v;

    listArr.nodeArr[edge[0].u] = 0;
    listArr.nodeArr[nodeNum] = edgeNum;
    for(int i = nodeNum; i > 0; i--){
        if(listArr.nodeArr[i-1] == -1)
            listArr.nodeArr[i-1] = listArr.nodeArr[i];
    }
}

