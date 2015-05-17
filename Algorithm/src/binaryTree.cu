#include "binaryTree.h"
#include <cstdio>

__host__ __device__ int nearestLessPowOf2(int num){
    int ans = 1;
    for(; ans < num; ans *= 2);
    if(ans==1) return 1;
    return ans/2;
}

__global__ void sumTriangle(long long *triNum, int entryNum){
/*    if(threadIdx.x==0){
        for(int i = 0; i < entryNum; i++){
            printf("triNum[%d]=%lld\n", i, triNum[i]);
        }
    }*/
//    sum(tmp, entryNum, blockDim.x);
    triNum[0] = linearSum(triNum, entryNum);
}

__device__ void binaryTreeSum(long long *triNum, int entryNum, int bound){
    int aliveBound = bound;
    while(aliveBound >= 1){
        if(threadIdx.x < aliveBound){
            triNum[threadIdx.x] += triNum[threadIdx.x+aliveBound];
            triNum[threadIdx.x+aliveBound] = 0;
        }
        aliveBound /= 2;
        __syncthreads();
    }
}

__device__ long long linearSum(long long *triNum, int entryNum){
    long long tmp = 0;
    for(int i = 0; i < entryNum; i++)
        tmp += triNum[i];
    return tmp;
}
