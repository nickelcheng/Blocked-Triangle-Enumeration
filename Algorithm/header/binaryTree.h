#ifndef __BINARY_TREE_H__
#define __BINARY_TREE_H__

__host__ __device__ int nearestLessPowOf2(int num);
__global__ void sumTriangle(long long *triNum, int entryNum);
__device__ void binaryTreeSum(long long *triNum, int entryNum, int bound);
__device__ long long linearSum(long long *triNum, int entryNum);

#endif
