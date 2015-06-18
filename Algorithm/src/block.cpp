#include "block.h"
#include "io.h"
#include "list.h"
#include "tool.h"
#include<algorithm>
#include<cstdio>

int initBlock(const vector< Edge > &edge, int nodeNum, int blockSize, vector< Matrix > &block){
    int blockDim = averageCeil(nodeNum, blockSize);
    vector< int* > blockEdgeNum(blockDim);
    countBlockEdgeNum(edge, blockDim, blockSize, blockEdgeNum);

    for(int i = 0; i < blockDim; i++){
        if(blockEdgeNum[i][i] > EDGE_NUM_LIMIT)
            fprintf(stderr, "error, EDGE_NUM_LIMIT(%d) too small or blockSize(%d) too large\n", EDGE_NUM_LIMIT, blockSize);
    }

    int *newBlockID = new int[blockDim];
    int newBlockDim = integrateBlock(blockEdgeNum, blockDim, newBlockID);
    splitBlock(edge, newBlockID, blockSize, newBlockDim, block);

    for(int i = 0; i < blockDim; i++)
        delete [] blockEdgeNum[i];
    delete [] newBlockID;

    return newBlockDim;
}

void countBlockEdgeNum(const vector< Edge > &edge, int blockDim, int blockSize, vector< int* > &blockEdgeNum){
    for(int i = 0; i < blockDim; i++){
        blockEdgeNum[i] = new int [blockDim];
        for(int j = i; j < blockDim; j++){
            blockEdgeNum[i][j] = 0;
        }
    }

    vector< Edge >::const_iterator it = edge.begin();
    for(; it != edge.end(); ++it){
        int u = it->u, v = it->v;
        int ublock = u / blockSize;
        int vblock = v / blockSize;
        blockEdgeNum[ublock][vblock]++;
    }
}

int integrateBlock(const vector< int* > &blockEdgeNum, int blockDim, int *newBlockID){
    int currEdgeNum = blockEdgeNum[0][0];
    int currBlockID = 0;
    newBlockID[0] = 0;
    for(int b = 1; b < blockDim; b++){
        int addEdgeNum = 0;
        for(int i = 0; i <= b; i++)
            addEdgeNum += blockEdgeNum[i][b];
        if(currEdgeNum + addEdgeNum > EDGE_NUM_LIMIT){
            currEdgeNum = blockEdgeNum[b][b];
            newBlockID[b] = ++currBlockID;
        }
        else{
            currEdgeNum += addEdgeNum;
            newBlockID[b] = currBlockID;
        }
    }
    return currBlockID+1;
}

void splitBlock(const vector< Edge > &edge, const int* newBlockID, int blockSize, int blockDim, vector< Matrix > &block){
    block = vector< Matrix >(blockDim);
    for(int i = 0; i < blockDim; i++){
        block[i] = vector< Row >(blockDim);
    }
    vector< Edge >::const_iterator it = edge.begin();
    for(; it != edge.end(); ++it){
        int u = it->u, v = it->v;
        int ublock = newBlockID[u/blockSize];
        int vblock = newBlockID[v/blockSize];
        block[ublock][vblock].push_back(Edge(u, v));
    }
}

void sortBlock(vector< Matrix > &block, int blockDim){
    for(int i = 0; i < blockDim; i++){
        for(int j = i; j < blockDim; j++){
            std::sort(block[i][j].begin(), block[i][j].end());
        }
    }
}

void relabelBlock(vector< Edge > &edge, int blockSize, int uOffset, int vOffset){
    if(edge.empty()) return;
    int currUOffset = edge.front().u / blockSize;
    int currVOffset = edge.front().v / blockSize;
    if(uOffset == currUOffset && vOffset == currVOffset) return;

    int uDiff = (uOffset-currUOffset) * blockSize;
    int vDiff = (vOffset-currVOffset) * blockSize;
    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        e->u += uDiff;
        e->v += vDiff;
    }
}

