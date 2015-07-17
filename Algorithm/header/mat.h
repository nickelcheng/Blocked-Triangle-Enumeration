#ifndef __MAT_H__
#define __MAT_H__

#include "listArray.h"
#include "bitMat.h"

typedef unsigned int UI;

const int MAX_NODE_NUM_LIMIT = 10*1024;

#ifdef __NVCC__
extern __shared__ UI tile[];
__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, UC *oneBitNum, long long *triNum);
#endif

void mat(int device, const ListArray &edge, const BitMat &target);
void cpuCountMat(const ListArray &edge, const BitMat &target);

void gpuCountTriangleMat(const ListArray &edge, const BitMat &target);
void createOneBitNumTable(UC *oneBitNum, UC **d_oneBitNum);
DECORATE long long countOneBits(UI tar, unsigned char *oneBitNum);

#endif
