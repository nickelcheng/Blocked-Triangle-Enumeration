#ifndef __MAT_H__
#define __MAT_H__

#include "main.h"
#include "listArray.h"
#include "bitMat.h"
#include "threadHandler.h"

typedef unsigned int UI;
#define BIT_PER_ENTRY (sizeof(UI)*8)

#ifdef __NVCC__
extern __shared__ UI tile[];
__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, long long *triNum);
#endif

const int MAX_NODE_NUM_LIMIT = 10*1024;

void mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum,
    int threadNum=256, int blockNum=1024
);
long long cpuCountMat(const MatArg &matArg);
void *callGpuMat(void *arg);
void *callCpuMat(void *arg);

long long  gpuCountTriangleMat(const MatArg &matArg);
DECORATE long long countOneBits(UI tar);

#endif
