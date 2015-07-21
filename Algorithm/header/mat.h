#ifndef __MAT_H__
#define __MAT_H__

#include "listArray.h"
#include "bitMat.h"

const int MAX_NODE_NUM_LIMIT = 40*1024;

#ifdef __NVCC__
//extern __shared__ UC tile[];
extern __shared__ long long threadTriNum[];
__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, UC *oneBitNum, long long *triNum);
#endif

void mat(int device, const ListArray &edge, const ListArray &target, int width);
void cpuCountMat(const ListArray &edge, const BitMat &target);

void gpuCountTriangleMat(const ListArray &edge, const ListArray &target, int entryNum);
void createOneBitNumTable(UC *oneBitNum, UC **d_oneBitNum);
DECORATE long long countOneBits(UC tar, UC *oneBitNum);

#endif
