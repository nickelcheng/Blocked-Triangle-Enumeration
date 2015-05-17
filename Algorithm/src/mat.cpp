#include "mat.h"
#include "tool.h"
#include "solve.h"

#include <cstring>

#include<cstdio>

long long mat(int device, int nodeNum, vector< Edge > &edge, int threadNum, int blockNum){
    int entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    UI *mat = new UI[entryNum*nodeNum];
    UI mask[BIT_PER_ENTRY];
    long long triNum = 0;

    createMask(BIT_PER_ENTRY, mask);
    initMatrix(edge, mat, nodeNum, entryNum, mask);

    if(device == CPU)
        triNum = cpuCountMat(mat, entryNum, nodeNum);

    else
        triNum = gpuCountTriangleMat(mat, entryNum, nodeNum, threadNum, blockNum);

    delete [] mat;

    return triNum;
}

long long cpuCountMat(UI *mat, int entryNum, int nodeNum){
    long long triNum = 0;
    for(int i = 0; i < nodeNum; i++){
        int offset = i / BIT_PER_ENTRY;
        int bit = i % BIT_PER_ENTRY;

        // iterator through each entry of the row
        for(int j = offset; j < entryNum; j++){
            // iterator through each bit
            UI content = mat[i*entryNum+j];
            int cst = j*BIT_PER_ENTRY;
            if(j == offset){
                for(int s = bit; s >= 0; s--, content/=2);
                cst = i+1;
            }
            for(int k = cst; content > 0; k++, content/=2){
                if(content % 2 == 1){ // edge(i, k) exists
                    for(int e = 0; e < entryNum; e++)
                        triNum += andList(mat, i, k, e, entryNum);
                }
            }
        }
    }
    return triNum/3;
}

void createMask(int maskNum, UI *mask){
    for(int i = 0; i < maskNum; i++){
        mask[i] = (UI)1 << i;
    }
}

void initMatrix(vector< Edge > &edge, UI *mat, int nodeNum, int entryNum, UI *mask){
    memset(mat, 0, sizeof(UI)*entryNum*nodeNum);
    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        setEdge(e->u, e->v);
        setEdge(e->v, e->u);
    }
}

