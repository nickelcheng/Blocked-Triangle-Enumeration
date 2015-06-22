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
    if(v < 0 || v >= nodeNum) return NULL;
    if(nodeArr[v] >= edgeNum) return NULL;
    return &edgeArr[nodeArr[v]];
}

void ListArray::integrate(const ListArray &a, ListArray &res) const{
    res.initArray(nodeNum+a.nodeNum, edgeNum+a.edgeNum);
    for(int i = 0; i < nodeNum; i++)
        res.nodeArr[i] = nodeArr[i];
    for(int i = 0, j = nodeNum; i < a.nodeNum; i++, j++)
        res.nodeArr[j] = a.nodeArr[i] + edgeNum;
    res.nodeArr[res.nodeNum] = res.edgeNum;

    for(int i = 0; i < edgeNum; i++)
        res.edgeArr[i] = edgeArr[i];
    for(int i = 0, j = edgeNum; i < a.edgeNum; i++, j++)
        res.edgeArr[j] = a.edgeArr[i];
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

__global__ void initNodeArr(int nodeNum, ListArray *listArr){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i < nodeNum; i+=threads)
        listArr->nodeArr[i] = -1;
}

__global__ void edge2listArr(const Edge *edge, int nodeNum, int edgeNum, ListArray *listArr){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i < edgeNum; i+=threads){
        listArr->edgeArr[i] = edge[i].v;
        if(i > 0){
            if(edge[i-1].u != edge[i].u)
                listArr->nodeArr[edge[i].u] = i;
        }
    }
    listArr->nodeArr[edge[0].u] = 0;
    listArr->nodeArr[nodeNum] = edgeNum;
}

__global__ void removeEmptyFlag(int nodeNum, ListArray *listArr){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i <= nodeNum; i+=threads){
        if(listArr->nodeArr[i] != -1 && i > 0){
            for(int j = i-1; j >= 0 && listArr->nodeArr[j]==-1; j--){
                listArr->nodeArr[j] = listArr->nodeArr[i];
            }
        }
    }
}
