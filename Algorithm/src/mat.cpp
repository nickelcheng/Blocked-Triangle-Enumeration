#include "mat.h"
#include "tool.h"
#include "solve.h"

#include <cstring>

#include<cstdio>

long long mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum
){
    ListArray edgeList;
    BitMat tarMat(nodeNum, entryNum);
    long long triNum = 0;

    edgeList.initArray(edge, edgeRange);
    tarMat.initMat(target);
    
    if(device == CPU || tarMat.nodeNum > MAX_NODE_NUM_LIMIT)
        triNum = cpuCountMat(edgeList, tarMat);

/*    else
        triNum = gpuCountTriangleMat(edgeMat, tarMat, threadNum, blockNum);*/

    return triNum;
}

long long cpuCountMat(const ListArray &edge, const BitMat &target){
    long long triNum = 0;
    // iterator through each edge
    int range = edge.getNodeNum();
    for(int u = 0; u < range; u++){

        const int *uNei = edge.neiStart(u);
        int uDeg = edge.getDeg(u);
        for(int i = 0; i < uDeg; i++){
            int v = uNei[i];
//            printf("intersect %d & %d\n", u, v);
            long long tmp = 0;
            for(int e = 0; e < target.entryNum; e++){
                UI e1 = target.getContent(u, e);
                UI e2 = target.getContent(v, e);
                tmp += countOneBits(e1 & e2);
//                if(u==5&&v==39) printf("%u %u\n", e1, e2);
            }
            triNum += tmp;
//            printf("%d & %d => %lld\n", u, v, tmp);
        }
    }
    return triNum;
}

