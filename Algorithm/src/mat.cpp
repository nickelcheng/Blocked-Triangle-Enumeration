#include "mat.h"
#include "tool.h"
#include "solve.h"

#include <cstring>

#include<cstdio>

long long mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum,
    int threadNum, int blockNum
){
    ListArray edgeList;
    BitMat tarMat(nodeNum, entryNum);
    long long triNum = 0;

    edgeList.initArray(edge, edgeRange);
    tarMat.initMat(target);
    
    if(device == CPU || tarMat.nodeNum > MAX_NODE_NUM_LIMIT)
        triNum = cpuCountMat(edgeList, tarMat);

    else
        triNum = gpuCountTriangleMat(edgeList, tarMat, threadNum, blockNum);

    return triNum;
}

long long cpuCountMat(const ListArray &edge, const BitMat &target){
    long long triNum = 0;
    for(int e = 0; e < target.entryNum; e++){
    // iterator through each edge
        int range = edge.getNodeNum();
        for(int u = 0; u < range; u++){

            const int *uNei = edge.neiStart(u);
            int uDeg = edge.getDeg(u);
            for(int i = 0; i < uDeg; i++){
                int v = uNei[i];
                UI e1 = target.getContent(u, e);
                UI e2 = target.getContent(v, e);
                long long tmp = countOneBits(e1 & e2);
                if(countOneBits(e1&e2)>0)
                triNum += tmp;
            }
        }
    }
    return triNum;
}

