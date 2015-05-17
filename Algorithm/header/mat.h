#ifndef __MAT_H__
#define __MAT_H__

#include "struct.h"
#include<cuda_runtime.h>

typedef unsigned int UI;
#define BIT_PER_ENTRY (sizeof(UI)*8)

#define setEdge(u,v) mat[u*entryNum+(v/BIT_PER_ENTRY)]|=mask[v%BIT_PER_ENTRY]

long long mat(int device, int nodeNum, vector< Edge > &edge, int threadNum=256, int blockNum=1024);
long long cpuCountMat(UI *mat, int entryNum, int nodeNum);
void createMask(int maskNum, UI *mask);
void initMatrix(vector< Edge > &edge, UI *mat, int nodeNum, int entryNum, UI *mask);

long long gpuCountTriangleMat(UI *mat, int entryNum, int nodeNum, int threadNum, int blockNum);
__global__ void gpuCountMat(UI *mat, int entryNum, int nodeNum, long long *triNum, int threadNum, int blockNum);
__host__ __device__ long long andList(UI *mat, int l1, int l2, int entry, int entryNum);
__host__ __device__ long long countOneBits(UI tar);

#endif
