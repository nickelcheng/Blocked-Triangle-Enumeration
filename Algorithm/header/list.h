#ifndef __LIST_H__
#define __LIST_H__

#include "listArray.h"

#ifdef __NVCC__
extern __shared__ int uAdj[];
__global__ void gpuCountList(const ListArray *edge, const ListArray *target, long long *triNum);
#endif

const int MAX_DEG_LIMIT = 10*1024;

void list(int device, const ListArray &edge, const ListArray &target);
void cpuCountList(const ListArray &edge, const ListArray &target);

void gpuCountTriangle(const ListArray &edge, const ListArray &target, int maxDeg);
DECORATE long long intersectList(int sz1, int sz2, const int *l1, const int *l2);

#endif
