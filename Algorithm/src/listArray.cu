#include "listArray.h"
#include<cstring>
#include<cstdio>

ListArray::~ListArray(void){
    delete [] nodeArr;
    delete [] edgeArr;
}

void ListArray::initArray(int n, int e){
    nodeNum = n, edgeNum = e;
    nodeArr = new int[sizeof(int)*(n+1)];
    edgeArr = new int[sizeof(int)*e];
}

int ListArray::getMaxDegree(void) const{
    int mmax = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = nodeArr[i+1] - nodeArr[i];
        if(deg > mmax) mmax = deg;
    }
    return mmax;
}

DECORATE  int ListArray::getDeg(int v) const{
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
