#include "mat.h"
#include "binaryTree.h"
#include "timer.h"
#include <cstdio>

void gpuCountTriangleMat(const ListArray &edge, const ListArray &target, int entryNum){
    extern UC *d_oneBitNum;
    long long *d_triNum, ans;

    ListArray *d_edge;
    int *d_edgeArr, *d_nodeArr;
    // copy edge to device
    cudaMalloc((void**)&d_edge, sizeof(ListArray));
    cudaMemcpy(d_edge, &edge, sizeof(ListArray), H2D);
    // edge.nodeArr
    cudaMalloc((void**)&d_nodeArr, sizeof(int)*(edge.nodeNum+1));
    cudaMemcpy(d_nodeArr, edge.nodeArr, sizeof(int)*(edge.nodeNum+1), H2D);
    cudaMemcpy(&(d_edge->nodeArr), &d_nodeArr, sizeof(int*), H2D);
    // edge.edgeArr
    cudaMalloc((void**)&d_edgeArr, sizeof(int)*edge.edgeNum);
    cudaMemcpy(d_edgeArr, edge.edgeArr, sizeof(int)*edge.edgeNum, H2D);
    cudaMemcpy(&(d_edge->edgeArr), &d_edgeArr, sizeof(int*), H2D);

//    timerInit(1)
//    timerStart(0)
    BitMat *d_tarMat;
    UI *d_mat;
//    cListArr2BitMat(target, &d_tarMat, &d_mat, entryNum);
    gListArr2BitMat(target, &d_tarMat, &d_mat, entryNum);
//    pListArr2BitMat(target, &d_tarMat, &d_mat, entryNum);
//    timerEnd("gpu list->bitmat", 0)

    extern int blockNum, threadNum;
    cudaMalloc((void**)&d_triNum, sizeof(long long)*blockNum);

//    timerStart(0)
//    int smSize = target.nodeNum*sizeof(UI);
//    gpuCountMat<<< blockNum, threadNum, smSize >>>(d_edge, d_tarMat, d_oneBitNum, d_triNum);
    int avgDeg = edge.edgeNum / edge.nodeNum;
    while(threadNum > avgDeg/8) threadNum -= 32;
    if(threadNum < 32) threadNum = 32;
    if(blockNum > entryNum) blockNum = entryNum;
    int smSize = threadNum*sizeof(long long);
    printf("block %d, thread %d, sm %d\n", blockNum, threadNum, smSize);
    gpuCountMat<<< blockNum, threadNum, smSize >>>(d_edge, d_tarMat, d_oneBitNum, d_triNum);
    sumTriangle<<< 1, 1 >>>(d_triNum, blockNum);
    cudaMemcpy(&ans, d_triNum, sizeof(long long), D2H);
//    timerEnd("mat count", 0)

    cudaFree(d_edge);
    cudaFree(d_edgeArr);
    cudaFree(d_nodeArr);
    cudaFree(d_triNum);
    cudaFree(d_tarMat);
    cudaFree(d_mat);

    extern long long triNum;
    triNum += ans;
}

__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, UC *oneBitNum, long long *triNum){

    int bound = nearestLessPowOf2(blockDim.x);

    triNum[blockIdx.x] = 0;

    for(int e = blockIdx.x; e < target->entryNum; e += gridDim.x){

        // move tile area to shared memory
        int offset = e * target->nodeNum;
/*        for(int i = threadIdx.x; i < target->nodeNum; i++){
            tile[i] = target->mat[offset+i];
        }
        __syncthreads();*/

        // count triangle number
        threadTriNum[threadIdx.x] = 0;
        // iterator through each edge (u, v)
        int range = edge->nodeNum;
        for(int u = 0; u < range; u++){
            int uDeg = edge->getDeg(u);
            if(uDeg > 0){
                const int *uNei = edge->neiStart(u);
                for(int i = threadIdx.x; i < uDeg; i += blockDim.x){
                    int v = uNei[i];
                    UI e1 = target->mat[offset+u];
                    UI e2 = target->mat[offset+v];
                    threadTriNum[threadIdx.x] += countOneBits(e1 & e2, oneBitNum);
//                    threadTriNum[threadIdx.x] += countOneBits(tile[u]&tile[v], oneBitNum);
                }
            }
        }
        __syncthreads();

/*        binaryTreeSum(threadTriNum, blockDim.x, bound);
        if(threadIdx.x==0)
            triNum[blockIdx.x] += threadTriNum[0];*/

        if(threadIdx.x==0)
            triNum[blockIdx.x] += linearSum(threadTriNum, blockDim.x);

        __syncthreads();
    }
}

void createOneBitNumTable(UC *oneBitNum, UC **d_oneBitNum){
    cudaMalloc((void**)d_oneBitNum, sizeof(UC)*BIT_NUM_TABLE_SIZE);
    for(int i = 0; i < BIT_NUM_TABLE_SIZE; i++){
        int ans = 0;
        for(int num = i; num > 0; ans++){
            num &= (num-1);
        }
        oneBitNum[i] = (UC)ans;
    }
    cudaMemcpy(*d_oneBitNum, oneBitNum, sizeof(UC)*BIT_NUM_TABLE_SIZE, H2D);
}

DECORATE long long countOneBits(UI tar, UC *oneBitNum){
    long long ones = 0;
    for(; tar; tar/=BIT_NUM_TABLE_SIZE)
        ones += oneBitNum[tar % BIT_NUM_TABLE_SIZE];
//    for(; tar; tar/=2)
//        ones += tar % 2;
    return ones;
}

