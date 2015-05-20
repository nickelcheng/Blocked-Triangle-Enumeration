#include "block.h"
#include "io.h"
#include "list.h"
#include<algorithm>

void initBlock(int blockDim, vector< Matrix > &block){
    for(int i = 0; i < blockDim; i++)
        block[i] = vector< Row >(blockDim);
}

void splitBlock(int blockSize, vector< Matrix > &block, vector< Edge > &edge){
    vector< Edge >::iterator it = edge.begin();
    for(; it != edge.end(); ++it){
        int u = it->u, v = it->v;
        int ublock = u / blockSize;
        int vblock = v / blockSize;
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

