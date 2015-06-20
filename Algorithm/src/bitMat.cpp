#include "bitMat.h"
#include <cstring>
#include <cstdio>
#include <omp.h>

UI BitMat::mask[BIT_PER_ENTRY];

void BitMat::createMask(void){
    for(int i = 0; i < (int)BIT_PER_ENTRY; i++)
        mask[i] = (UI)1 << i;
}

BitMat::~BitMat(void){
    delete [] mat;
}

void BitMat::initMat(const ListArray &edge, int entry){
    nodeNum = edge.nodeNum;
    entryNum = entry;
    mat = new UI[sizeof(UI)*entryNum*nodeNum];
    memset(mat, 0, sizeof(UI)*entryNum*nodeNum);

    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++){
        for(int j = edge.nodeArr[i]; j < edge.nodeArr[i+1]; j++){
            setEdge(i, edge.edgeArr[j]);
        }
    }
}

UI BitMat::getContent(int node, int entry) const{
    return mat[entry*nodeNum+node];
}

void BitMat::setEdge(int u, int v){
    int row = v / BIT_PER_ENTRY, col = u;
    int bit = v % BIT_PER_ENTRY;
    #pragma omp critical
    {
    mat[row*nodeNum+col] |= mask[bit];
    }
}


