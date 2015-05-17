#include "mat.h"
#include "binaryTree.h"

#include<cstdio>

long long gpuCountTriangleMat(UI *mat, int entryNum, int nodeNum, int threadNum, int blockNum){
    long long *d_triNum, triNum;
    UI *d_mat;

    cudaMalloc((void**)&d_triNum, sizeof(long long)*blockNum);
    cudaMalloc((void**)&d_mat, sizeof(UI)*entryNum*nodeNum);
    cudaMemcpy(d_mat, mat, sizeof(UI)*entryNum*nodeNum, cudaMemcpyHostToDevice);

    gpuCountMat<<< blockNum, threadNum >>>(d_mat, entryNum, nodeNum, d_triNum, threadNum, blockNum);
    cudaDeviceSynchronize();

    sumTriangle<<< 1, 1 >>>(d_triNum, blockNum);
    cudaDeviceSynchronize();
    cudaMemcpy(&triNum, d_triNum, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_triNum);
    cudaFree(d_mat);

    return triNum/6;
}

__global__ void gpuCountMat(UI *mat, int entryNum, int nodeNum, long long *triNum, int threadNum, int blockNum){
    __shared__ long long threadTriNum[1024];
    int bound = nearestLessPowOf2(blockDim.x);

    if(threadIdx.x==0)
        triNum[blockIdx.x] = 0;
    __syncthreads();

    for(int e = blockIdx.x; e < entryNum; e += gridDim.x){

        // count triangle number
        threadTriNum[threadIdx.x] = 0;
        for(int i = threadIdx.x; i < nodeNum; i += blockDim.x){
            // iterator through each entry of the row
            for(int j = 0; j < entryNum; j++){
                // iterator through each bit
                UI content = mat[i*entryNum+j];
                for(int k = j*BIT_PER_ENTRY; content > 0; k++, content/=2){
                    if(content % 2 == 1){ // edge(i, k) exists
                        threadTriNum[threadIdx.x] += andList(mat, i, k, e, entryNum);
                    }
                }
            }
        }
        __syncthreads();

        binaryTreeSum(threadTriNum, blockDim.x, bound);
        if(threadIdx.x==0){
            triNum[blockIdx.x] += threadTriNum[0];
        }

/*        if(threadIdx.x==0)
            triNum[blockIdx.x] += linearSum(threadTriNum, blockDim.x);*/

        __syncthreads();
    }
}

__host__ __device__ long long andList(UI *mat, int l1, int l2, int entry, int entryNum){
    long long triNum = 0;
    UI result = mat[l1*entryNum+entry] & mat[l2*entryNum+entry];
    triNum = countOneBits(result);
    return triNum;
}

__host__ __device__ long long countOneBits(UI tar){
    long long ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}

