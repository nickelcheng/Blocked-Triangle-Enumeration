#ifndef __LIST_H__
#define __LIST_H__

#include "main.h"
#include "listArray.h"
#include "threadHandler.h"

#ifdef __NVCC__
extern __shared__ int uAdj[];
__global__ void gpuCountList(const ListArray *edge, const ListArray *target, long long *triNum);
#endif

const int MAX_DEG_LIMIT = 10*1024;

void list(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum,
    int threadNum=256, int blockNum=1024
);
long long cpuCountList(const ListArg &listArg);
void *callCpuList(void *arg);
void *callGpuList(void *arg);

long long gpuCountTriangle(const ListArg &listArg);
DECORATE long long intersectList(int sz1, int sz2, const int *l1, const int *l2);

#endif
