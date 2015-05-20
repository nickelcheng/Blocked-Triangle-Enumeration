#ifndef __MAT_H__
#define __MAT_H__

#include "struct.h"
#include "listArray.h"
#include "bitMat.h"
#include<cuda_runtime.h>

typedef unsigned int UI;
#define BIT_PER_ENTRY (sizeof(UI)*8)

extern __shared__ UI tile[];
const int MAX_NODE_NUM_LIMIT = 10*1024;

long long mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum,
    int threadNum=256, int blockNum=1024
);
long long cpuCountMat(const ListArray &edge, const BitMat &target);

long long gpuCountTriangleMat(const ListArray &edge, const BitMat &target, int threadNum, int blockNum);
__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, long long *triNum);
__host__ __device__ long long countOneBits(UI tar);

#endif
