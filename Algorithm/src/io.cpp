#include "io.h"
#include<cstdio>

void inputEdge(const char *inFile, vector< Edge > &edge){
    FILE *fp = fopen(inFile, "r");
    int u, v;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        edge.push_back(Edge(u,v));
    }
    fclose(fp);
}

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
        if(u < v) block[ublock][vblock].push_back(Edge(u, v));
        else block[vblock][ublock].push_back(Edge(v, u));
    }
}

void relabelBlock(int blockSize, int blockDim, vector< Matrix > &block){
    for(int i = 0; i < blockDim; i++){
        // dialog blocks: id 0~blockSize-1
        relabel(-blockSize*i, block[i][i]);
        for(int j = i+1; j < blockDim; j++){
            // other blocks: id blockSize ~ 2*blockSize-1
            relabel(-blockSize*(j-1), block[i][j]);
        }
    }
}

void relabel(int offset, vector< Edge > &edge){
    if(offset == 0) return;
    vector< Edge >::iterator e;
    for(e = edge.begin(); e != edge.end(); ++e){
        e->u += offset;
        e->v += offset;
    }
}
