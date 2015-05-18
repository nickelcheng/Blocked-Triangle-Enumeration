#ifndef __LIST_H__
#define __LIST_H__

#include "struct.h"
#include <cuda_runtime.h>

typedef struct listNode{
    vector< int > nei;
    int degree(void) const{
        return (int)nei.size();
    }
} ListNode;

extern __shared__ int uAdj[];

const int MAX_DEG_LIMIT = 10*1024;

long long forward(int device, int nodeNum, vector< Edge > edge, int threadNum=256, int blockNum=1024);
long long cpuCountList(int *nodeArr, int *edgeArr, int nodeNum);
void initArray(int *nodeArr, int *edgeArr, vector< Edge > &edge, int nodeNum, int edgeNum);
int getMaxDeg(int *nodeArr, int nodeNum);

long long gpuCountTriangle(int *nodeArr, int *edgeArr, int nodeNum, int edgeNum, int maxDeg, int threadNum, int blockNum);
__global__ void gpuCountList(int *nodeArr, int *edgeArr, int nodeNum, long long *triNum);
__host__ __device__ int getDeg(int *nodeArr, int v);
__host__ __device__ long long intersectList(int sz1, int sz2, int *l1, int *l2);

#endif
