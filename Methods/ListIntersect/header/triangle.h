#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include "ds.h"
#include<cuda_runtime.h>

extern __shared__ int shared[]; // adj[maxDeg], threadTriNum[threadNum]

int cpuCountTriNum(vector< Node > &node);
void listCopy(int **offset, int **edgeV, int edgeNum, vector< Node > &node);
void listCopyToDevice(vector< Node > &node, int edgeNum, void** d_offset, void** d_edgeV);
__global__ void gpuCountTriNum(int *offset, int *edgeV, int *triNum, int nodeNum);
__host__ __device__ int intersectList(int sz1, int sz2, int *l1, int *l2);


#endif
