#include "listArray.h"
#include<cstring>
#include<cstdio>

ListArray::~ListArray(void){
    delete [] nodeArr;
    delete [] edgeArr;
}

void ListArray::initArray(const vector< Edge > &edge, int n){
    nodeNum = n, edgeNum = (int)edge.size();
    nodeArr = new int[sizeof(int)*(n+1)];
    edgeArr = new int[sizeof(int)*edgeNum];
    memset(nodeArr, -1, sizeof(int)*(nodeNum+1));

    for(int i = 0; i < edgeNum-1; i++){
        edgeArr[i] = edge[i].v;
        if(edge[i].u != edge[i+1].u){
            nodeArr[edge[i+1].u] = i+1;
        }
    }
    edgeArr[edgeNum-1] = edge[edgeNum-1].v;

    nodeArr[edge[0].u] = 0;
    nodeArr[nodeNum] = edgeNum;
    for(int i = nodeNum; i > 0; i--){
        if(nodeArr[i-1] == -1) nodeArr[i-1] = nodeArr[i];
    }
}

int ListArray::getMaxDegree(void) const{
    int mmax = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = nodeArr[i+1] - nodeArr[i];
        if(deg > mmax) mmax = deg;
    }
    return mmax;
}

__host__ __device__ int ListArray::getDeg(int v) const{
    if(v < 0 || v >= nodeNum) return 0;
    return nodeArr[v+1] - nodeArr[v];
}

DECORATE int ListArray::getNodeNum() const{
    return nodeNum;
}

DECORATE const int* ListArray::neiStart(int v) const{
    if(v < 0 || v >= nodeNum) return 0;
    return &edgeArr[nodeArr[v]];
}

DECORATE void ListArray::print(void) const{
    printf("node:");
    for(int i = 0; i <= nodeNum; i++){
        printf(" %d", nodeArr[i]);
    }
    printf("\nedge:");
    for(int i = 0; i < edgeNum; i++){
        printf(" %d", edgeArr[i]);
    }
    printf("\n");
}
