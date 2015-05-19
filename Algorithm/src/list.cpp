#include "list.h"
#include "listArray.h"
#include "reorder.h"
#include "solve.h"
#include <cstring>
#include <algorithm>

#include<cstdio>

using namespace std;

long long list(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum,
    int threadNum, int blockNum
){
    ListArray edgeList, tarList;
    long long triNum = 0;

    edgeList.initArray(edge, edgeRange);
    tarList.initArray(target, nodeNum);

    int maxDeg = tarList.getMaxDegree();

    if(device == CPU || maxDeg > MAX_DEG_LIMIT)
        triNum = cpuCountList(edgeList, tarList);

    else
        triNum = gpuCountTriangle(edgeList, tarList, maxDeg, threadNum, blockNum);

    return triNum;
}

long long cpuCountList(const ListArray &edge, const ListArray &target){
    long long triNum = 0;
    // iterator through each edge (u, v)
    int range = edge.getNodeNum();
    for(int u = 0; u < range; u++){
        int uLen = target.getDeg(u);
        const int *uList = target.neiStart(u);

        const int *uNei = edge.neiStart(u);
        int uDeg = edge.getDeg(u);
        for(int i = 0; i < uDeg; i++){
            int v = uNei[i];
            int vLen = target.getDeg(v);
            const int *vList = target.neiStart(v);
            // intersect u list and v list in target
            triNum += intersectList(uLen, vLen, uList, vList);
        }
    }
    return triNum;
}

