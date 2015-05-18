#include "list.h"
#include "binaryTree.h"

long long gpuCountTriangle(int *nodeArr, int *edgeArr, int nodeNum, int edgeNum, int maxDeg, int threadNum, int blockNum){
    long long *d_triNum, triNum;
    int *d_nodeArr, *d_edgeArr;

    cudaMalloc((void**)&d_triNum, sizeof(long long)*blockNum);
    cudaMalloc((void**)&d_nodeArr, sizeof(int)*(nodeNum+1));
    cudaMalloc((void**)&d_edgeArr, sizeof(int)*edgeNum);
    cudaMemcpy(d_nodeArr, nodeArr, sizeof(int)*(nodeNum+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeArr, edgeArr, sizeof(int)*edgeNum, cudaMemcpyHostToDevice);

    int smSize = maxDeg*sizeof(int);
    gpuCount<<< blockNum, threadNum, smSize >>>(d_nodeArr, d_edgeArr, nodeNum, d_triNum);
    cudaDeviceSynchronize();

    sumTriangle<<< 1, 1 >>>(d_triNum, blockNum);
    cudaDeviceSynchronize();
    cudaMemcpy(&triNum, d_triNum, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_triNum);
    cudaFree(d_nodeArr);
    cudaFree(d_edgeArr);

    return triNum;
}

__global__ void gpuCount(int *nodeArr, int *edgeArr, int nodeNum, long long *triNum){
    __shared__ long long threadTriNum[1024];
    int bound = nearestLessPowOf2(blockDim.x);

    triNum[blockIdx.x] = 0;

    for(int u = blockIdx.x; u < nodeNum; u += gridDim.x){
        int uDeg = getDeg(nodeArr, u);
        
        // move node u's adj list to shared memory
        int offset = nodeArr[u];
        for(int i = threadIdx.x; i < uDeg; i += blockDim.x){
            uAdj[i] = edgeArr[offset+i];
        }
        __syncthreads();

        // counting triangle number
        threadTriNum[threadIdx.x] = 0;
        for(int i = threadIdx.x; i < uDeg; i += blockDim.x){
            int v = uAdj[i];;
            int vDeg = getDeg(nodeArr, v);
            threadTriNum[threadIdx.x] += intersectList(uDeg, vDeg, uAdj, &edgeArr[nodeArr[v]]);
        }
        __syncthreads();

        binaryTreeSum(threadTriNum, blockDim.x, bound);
        if(threadIdx.x==0)
            triNum[blockIdx.x] += threadTriNum[0];

/*        if(threadIdx.x==0)
            triNum[u] = linearSum(threadTriNum, blockDim.x);*/

        __syncthreads();
    }
}

__global__ void initTriNum(long long *triNum, int entryNum){
    for(int i = threadIdx.x; i < entryNum; i += blockDim.x)
        triNum[i] = 0;
}

__host__ __device__ int getDeg(int *nodeArr, int v){
    return nodeArr[v+1] - nodeArr[v];
}

__host__ __device__ long long intersectList(int sz1, int sz2, int *l1, int *l2){
    long long triNum = 0;
    for(int i = sz1-1, j = sz2-1; i >= 0 && j >= 0;){
        if(l1[i] > l2[j]) i--;
        else if(l1[i] < l2[j]) j--;
        else{
            i--, j--;
            triNum++;
        }
    }
    return triNum;
}
