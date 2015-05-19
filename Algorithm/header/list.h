#ifndef __LIST_H__
#define __LIST_H__

#include "struct.h"
#include "listArray.h"
#include <cuda_runtime.h>

typedef struct listNode{
    vector< int > nei;
    int degree(void) const{
        return (int)nei.size();
    }
} ListNode;

extern __shared__ int uAdj[];

const int MAX_DEG_LIMIT = 10*1024;

long long list(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum,
    int threadNum=256, int blockNum=1024
);
long long cpuCountList(const ListArray &edge, const ListArray &target);

long long gpuCountTriangle(
    const ListArray &edge, const ListArray &target,
    int maxDeg, int threadNum, int blockNum
);
__global__ void gpuCountList(const ListArray *edge, const ListArray *target, long long *triNum);
__host__ __device__ long long intersectList(int sz1, int sz2, const int *l1, const int *l2);

#endif
