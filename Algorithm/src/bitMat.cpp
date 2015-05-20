#include "bitMat.h"
#include "tool.h"
#include <cstring>
#include <cstdio>

BitMat::BitMat(int node){
    nodeNum = node;
    entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    mat = new UI[sizeof(UI)*node*entryNum];
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

EdgeMat::EdgeMat(int node) : BitMat(node){
}

UI EdgeMat::getContent(int node, int entry) const{
    return mat[node*entryNum+entry];
}

void EdgeMat::setEdge(int u, int v, const UI *mask){
    int row = u, col = v / BIT_PER_ENTRY;
    int bit = v % BIT_PER_ENTRY;
    mat[row*entryNum+col] |= mask[bit];
}

TarMat::TarMat(int node) : BitMat(node){
}

UI TarMat::getContent(int node, int entry) const{
    return mat[entry*nodeNum+node];
}

void TarMat::setEdge(int u, int v, const UI *mask){
    int row = v / 32, col = u;
    int bit = v % BIT_PER_ENTRY;
    mat[row*nodeNum+col] |= mask[bit];
}

