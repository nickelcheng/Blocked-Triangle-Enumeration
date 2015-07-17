#ifndef __MAT_H__
#define __MAT_H__

#include "listArray.h"
#include "bitMat.h"
#include "threadHandler.h"

typedef unsigned int UI;

const int MAX_NODE_NUM_LIMIT = 10*1024;

#ifdef __NVCC__
extern __shared__ UI tile[];
__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, long long *triNum);
#endif

void mat(int device, const ListArray &edge, const BitMat &target);
void cpuCountMat(const MatArg &matArg);

void gpuCountTriangleMat(const MatArg &matArg);
void createOneBitNumTable(unsigned char *oneBitNum);
DECORATE long long countOneBits(UI tar, unsigned char *oneBitNum);

#endif
