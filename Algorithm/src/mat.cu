#include "mat.h"
#include "binaryTree.h"

void gpuCountTriangleMat(const MatArg &matArg){
    const ListArray &edge = *(matArg.edge);
    const BitMat &target = *(matArg.target);

    long long *d_triNum, ans;
    ListArray *d_edge;
    BitMat *d_target;
    int *d_edgeArr, *d_nodeArr, *d_mat;

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

    delete &target;

    int smSize = target.nodeNum*sizeof(UI);
    gpuCountMat<<< blockNum, threadNum, smSize >>>(d_edge, d_target, d_triNum);
    sumTriangle<<< 1, 1 >>>(d_triNum, blockNum);
    cudaMemcpy(&ans, d_triNum, sizeof(long long), D2H);

    cudaFree(d_triNum);
    cudaFree(d_edge);
    cudaFree(d_target);
    cudaFree(d_edgeArr);
    cudaFree(d_nodeArr);
    cudaFree(d_mat);

    extern long long triNum;
    extern pthread_mutex_t lock;
    pthread_mutex_lock(&lock);
    triNum += ans;
    pthread_mutex_unlock(&lock);
}

__global__ void gpuCountMat(const ListArray *edge, const BitMat *target, long long *triNum){
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
                    threadTriNum[threadIdx.x] += countOneBits(tile[u]&tile[v]);
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

DECORATE long long countOneBits(UI tar){
    long long ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}

