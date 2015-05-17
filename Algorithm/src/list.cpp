#include "list.h"
#include "reorder.h"
#include "solve.h"
#include <cstring>
#include <algorithm>

using namespace std;

long long forward(int device, int nodeNum, vector< Edge > &edge, int threadNum, int blockNum){
    int edgeNum = (int)edge.size();
    int *nodeArr = new int[sizeof(int)*(nodeNum+1)];
    int *edgeArr = new int[sizeof(int)*edgeNum];
    long long triNum = 0;
    int maxDeg;

    forwardReorder(nodeNum, edge);
    initArray(nodeArr, edgeArr, edge, nodeNum, edgeNum);
    maxDeg = getMaxDeg(nodeArr, nodeNum);

    if(device == CPU || maxDeg > MAX_DEG_LIMIT)
        triNum = cpuCount(nodeArr, edgeArr, nodeNum);

    else
        triNum = gpuCountTriangle(nodeArr, edgeArr, nodeNum, edgeNum, maxDeg, threadNum, blockNum);

    delete [] nodeArr;
    delete [] edgeArr;

    return triNum;
}

long long cpuCount(int *nodeArr, int *edgeArr, int nodeNum){
    long long triNum = 0;
    for(int u = 0; u < nodeNum; u++){
        int uDeg = getDeg(nodeArr, u);
        long long tmp = 0;
        for(int i = nodeArr[u]; i < nodeArr[u+1]; i++){
            int v = edgeArr[i];
            int vDeg = getDeg(nodeArr, v);
            tmp += intersectList(uDeg, vDeg, &edgeArr[nodeArr[u]], &edgeArr[nodeArr[v]]);
        }
        triNum += tmp;
    }
    return triNum;
}

void initArray(int *nodeArr, int *edgeArr, vector< Edge > &edge, int nodeNum, int edgeNum){
    sort(edge.begin(), edge.end());
    memset(nodeArr, -1, sizeof(int)*(nodeNum+1));

    edge.push_back(Edge(nodeNum, -1));
    for(int i = 0; i < edgeNum; i++){
        edgeArr[i] = edge[i].v;
        if(edge[i].u != edge[i+1].u){
            nodeArr[edge[i+1].u] = i+1;
        }
    }
    edge.pop_back();

    nodeArr[edge[0].u] = 0;
    for(int i = nodeNum; i > 0; i--){
        if(nodeArr[i-1] == -1) nodeArr[i-1] = nodeArr[i];
    }
}

int getMaxDeg(int *nodeArr, int nodeNum){
    int mmax = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = nodeArr[i+1] - nodeArr[i];
        if(deg > mmax) mmax = deg;
    }
    return mmax;
}

