#include "bitMat.h"
#include "tool.h"
#include <cstring>
#include <cstdio>

BitMat::BitMat(int node, int entry){
    nodeNum = node;
    entryNum = entry;
    mat = new UI[sizeof(UI)*entryNum*nodeNum];
    memset(mat, 0, sizeof(UI)*entryNum*nodeNum);
}

BitMat::~BitMat(void){
    delete [] mat;
}

void BitMat::initMat(const vector< Edge > &edge){
    UI mask[BIT_PER_ENTRY];
    for(int i = 0; i < (int)BIT_PER_ENTRY; i++)
        mask[i] = (UI)1 << i;

    vector< Edge >::const_iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        setEdge(e->u, e->v, mask);
    }
}

UI BitMat::getContent(int node, int entry) const{
    return mat[entry*nodeNum+node];
}

void BitMat::setEdge(int u, int v, const UI *mask){
    int row = v / 32, col = u;
    int bit = v % BIT_PER_ENTRY;
    mat[row*nodeNum+col] |= mask[bit];
}


