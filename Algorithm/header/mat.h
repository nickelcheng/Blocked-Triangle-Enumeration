#ifndef __MAT_H__
#define __MAT_H__

#include "listArray.h"
#include "bitMat.h"
#include "threadHandler.h"
#include <thrust/device_vector.h>

typedef unsigned int UI;

using thrust::device_vector;

#ifdef __NVCC__
extern __shared__ UI tile[];
__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, long long *triNum);
#endif

const int MAX_NODE_NUM_LIMIT = 10*1024;

void mat(int device, const ListArray &edge, const BitMat &target);
void cpuCountMat(const MatArg &matArg);

void gpuCountTriangleMat(const MatArg &matArg);
DECORATE long long countOneBits(UI tar);

#endif
