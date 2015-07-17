#include "mat.h"
#include "binaryTree.h"
#include "timer.h"

//__constant__ unsigned char d_oneBitNum[BIT_NUM_TABLE_SIZE];

void gpuCountTriangleMat(const ListArray &edge, const BitMat &target){
    extern UC *d_oneBitNum;

    long long *d_triNum, ans;
    ListArray *d_edge;
    BitMat *d_target;
    int *d_edgeArr, *d_nodeArr, *d_mat;

    timerInit(1)

    timerStart(0)
    extern int blockNum, threadNum;
    if(blockNum > edge.nodeNum) blockNum = edge.nodeNum;
    cudaMalloc((void**)&d_triNum, sizeof(long long)*blockNum);

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

    // copy target to device
    cudaMalloc((void**)&d_target, sizeof(BitMat));
    cudaMemcpy(d_target, &target, sizeof(BitMat), H2D);
    // target.mat
    cudaMalloc((void**)&d_mat, sizeof(UI)*target.entryNum*target.nodeNum);
    cudaMemcpy(d_mat, target.mat, sizeof(UI)*target.entryNum*target.nodeNum, H2D);
    cudaMemcpy(&(d_target->mat), &d_mat, sizeof(UI*), H2D);
    timerEnd("copy", 0)

    int smSize = target.nodeNum*sizeof(UI);
    gpuCountMat<<< blockNum, threadNum, smSize >>>(d_edge, d_target, d_oneBitNum, d_triNum);
    sumTriangle<<< 1, 1 >>>(d_triNum, blockNum);
    cudaMemcpy(&ans, d_triNum, sizeof(long long), D2H);

    cudaFree(d_triNum);
    cudaFree(d_edge);
    cudaFree(d_target);
    cudaFree(d_edgeArr);
    cudaFree(d_nodeArr);
    cudaFree(d_mat);

    extern long long triNum;
    triNum += ans;
}

__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, UC *oneBitNum, long long *triNum){
    __shared__ long long threadTriNum[1024];
    int bound = nearestLessPowOf2(blockDim.x);

    triNum[blockIdx.x] = 0;

    for(int e = blockIdx.x; e < target->entryNum; e += gridDim.x){

        // move tile area to shared memory
        int offset = e * target->nodeNum;
        for(int i = threadIdx.x; i < target->nodeNum; i++){
            tile[i] = target->mat[offset+i];
        }
        __syncthreads();

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
/*                    UI e1 = target->getContent(u, e);
                    UI e2 = target->getContent(v, e);
                    threadTriNum[threadIdx.x] += countOneBits(e1 & e2);*/
                    threadTriNum[threadIdx.x] += countOneBits(tile[u]&tile[v], oneBitNum);
                }
            }
        }
        __syncthreads();

        binaryTreeSum(threadTriNum, blockDim.x, bound);
        if(threadIdx.x==0)
            triNum[blockIdx.x] += threadTriNum[0];

//        if(threadIdx.x==0)
//            triNum[blockIdx.x] += linearSum(threadTriNum, blockDim.x);

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

