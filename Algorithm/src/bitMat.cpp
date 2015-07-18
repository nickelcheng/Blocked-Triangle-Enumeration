#include "bitMat.h"
#include <cstring>

BitMat::~BitMat(void){
    delete [] mat;
}

void BitMat::initMat(const ListArray &edge, int entry){
    nodeNum = edge.nodeNum;
    entryNum = entry;
    mat = new UC[sizeof(UC)*entryNum*nodeNum];
    memset(mat, 0, sizeof(UC)*entryNum*nodeNum);

    for(int i = 0; i < nodeNum; i++){
        for(int j = edge.nodeArr[i]; j < edge.nodeArr[i+1]; j++){
            setEdge(i, edge.edgeArr[j]);
        }
    }
}

UC BitMat::getContent(int node, int entry) const{
    return mat[entry*nodeNum+node];
}

void BitMat::setEdge(int u, int v){
    extern UC mask[BIT_PER_ENTRY];
    int row = v / BIT_PER_ENTRY, col = u;
    int bit = v % BIT_PER_ENTRY;
    mat[row*nodeNum+col] |= mask[bit];
}
